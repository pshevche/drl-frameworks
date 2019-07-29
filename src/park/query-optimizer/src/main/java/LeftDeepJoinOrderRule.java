import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.prepare.CalcitePrepareImpl;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.metadata.RelMdUtil;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.mapping.Mappings;
import com.google.common.collect.ImmutableList;
import java.io.PrintWriter;
import java.util.*;

import static org.apache.calcite.util.mapping.Mappings.TargetMapping;

// new ones
import org.apache.calcite.plan.volcano.*;
import org.apache.calcite.rel.core.*;

// experimental
import org.apache.calcite.plan.RelOptUtil;
import java.util.concurrent.ThreadLocalRandom;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.*;

/**
 */
public class LeftDeepJoinOrderRule extends RelOptRule
{
  public static final LeftDeepJoinOrderRule INSTANCE =
      new LeftDeepJoinOrderRule(RelFactories.LOGICAL_BUILDER);

  // The keys represent a set of factors in the original vertices of the
  // QueryGraph. The values represent a sequence of edges (it's index at each
  // stage in QueryGraph.edges) that are chosen for the optimal memoized
  // ordering for the given set of factors. We can use these to reconstruct the
  // QueryGraph with a sequence of updateGraph steps for each
  // of the edge
  private HashMap<Set<Integer>, ArrayList<Integer>> memoizedBestJoins;

  /** Creates an LeftDeepJoinOrderRule. */
  public LeftDeepJoinOrderRule(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()), relBuilderFactory, null);
  }

  @Deprecated // to be removed before 2.0
  public LeftDeepJoinOrderRule(RelFactories.JoinFactory joinFactory,
      RelFactories.ProjectFactory projectFactory) {
    this(RelBuilder.proto(joinFactory, projectFactory));
  }

	/*
	 * We follow the algorithm described at:
	 * http://www.mathcs.emory.edu/~cheung/Courses/554/Syllabus/5-query-opt/dyn-prog-join2.html
	 */
  @Override
  public void onMatch(RelOptRuleCall call)
  {
    // Setting original expressions importance to 0, so our choice will be
    // chosen.
    RelNode orig = call.getRelList().get(0);
    call.getPlanner().setImportance(orig, 0.0);
    memoizedBestJoins = new HashMap<Set<Integer>, ArrayList<Integer>>();

    final MultiJoin multiJoinRel = call.rel(0);
    final RexBuilder rexBuilder = multiJoinRel.getCluster().getRexBuilder();
    final RelBuilder relBuilder = call.builder();
    //final MyMetadataQuery mq = QueryOptExperiment.getMetadataQuery();
    final MyMetadataQuery mq = MyMetadataQuery.instance();
    final LoptMultiJoin multiJoin = new LoptMultiJoin(multiJoinRel);

    // k = 1 case.
    for (int i = 0; i < multiJoin.getNumJoinFactors(); i++) {
       HashSet<Integer> factor = new HashSet<Integer>();
       factor.add(i);
			 // no edges have been chosen yet, so we add an empty list
			 memoizedBestJoins.put(factor, new ArrayList<Integer>());
    }

    ArrayList<Integer> initialVertexIdxs = new ArrayList<Integer>();
    for (int i = 0; i < multiJoin.getNumJoinFactors(); i++) {
      initialVertexIdxs.add(i);
    }

    // optimization: consider ab = ba.
    for (int k = 2; k < multiJoin.getNumJoinFactors()+1; k++) {
        List<Set<Integer>> res = new ArrayList<>();
        getSubsets(initialVertexIdxs, k, 0, new HashSet<Integer>(), res);
        for (Set<Integer> subset : res) {
            // the subset might already be memoized.
            double minCost = Double.POSITIVE_INFINITY;
            ArrayList<Integer> bestEdges = null;

            // basically, at the end of this iteration, we should have the current
            // subset entered into the memoized map.
            for (Integer r : subset) {
                HashSet<Integer> S_i = new HashSet<Integer>(subset);
                S_i.remove(r);
                //System.out.println("S_i is: " + S_i);
                ArrayList<Integer> optimalMemoizedEdges = memoizedBestJoins.get(S_i);
                if (optimalMemoizedEdges == null) {
                    // must not have been a valid join (e.g., there may not have been any
                    // edges between the given factors)
                    continue;
                }

                // create new QueryGraph and reconstruct the joins using the
                // memoizedFactors. Note: it is important to pass in a new copy of
                // the relBuilder because it maintains state, so that can not be
                // shared among multiple queryGraphs
                QueryGraph qg = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());
                for (Integer edge : optimalMemoizedEdges) {
                    qg.updateGraph(qg.edges.get(edge).factors.toArray());
                }
                double curCost = qg.costSoFar;
                // try to do the join represented by optimalMemoizedEdges AND the factor r.
                // Can only do this join if there is an edge connecting r with one of
                // the elements in S_i. This must be one of the unusedEdges.
                //List<QueryGraphUtils.Vertex> curVertexes = new ArrayList(optimalMemoizedEdges.curVertexes);
                //List<LoptMultiJoin2.Edge> curUnusedEdges = new ArrayList(optimalMemoizedEdges.curUnusedEdges);

                ImmutableBitSet.Builder neededFactorsBld = ImmutableBitSet.builder();
                neededFactorsBld.set(r);
                if (S_i.size() == 1) {
                    for (Integer s_i : S_i) {
                        neededFactorsBld.set(s_i);
                    }
                } else {
        //// FIXME: Not sure about this.
                    // the last vertex in curVertexes must be the right factor since we
                    // are doing left-deep joins
                    neededFactorsBld.set(qg.allVertexes.size() - 1);
                }
                ImmutableBitSet neededFactors = neededFactorsBld.build();

                // FIXME: this might not work if edge.factors have the latest
                // vertex rather than original indices of all the factors.
                // for vertices with more than one factor, we know the last vertex
                // added to curVertexes represents the best joined vertex so far
                // (...)
                Integer bestEdge = null;
                for (int edgeOrd = 0; edgeOrd < qg.edges.size(); edgeOrd+=1) {
                    // FIXME: do we need curEdge?
                    QueryGraph.Edge curEdge = qg.edges.get(edgeOrd);
                    if (curEdge.factors.equals(neededFactors)) {
                        bestEdge = edgeOrd;
                        break;
                    }
                }
                if (bestEdge == null) {
                    //System.out.println("no best edge!!!");
                    continue;
                }

                int [] factors = qg.edges.get(bestEdge).factors.toArray();
                curCost += qg.updateGraph(factors);
                if (curCost < minCost) {
                    minCost = curCost;
                    bestEdges = new ArrayList<Integer>(optimalMemoizedEdges);
                    bestEdges.add(bestEdge);
                }
            }

            if (bestEdges == null) continue;
            memoizedBestJoins.put(subset, bestEdges);
        }
    }
    // celebrate and add the relNode being developed so far from optRelNodes
    // to the set of equivalent nodes.
    QueryGraph qg = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());
    Set<Integer> allFactors = new HashSet<Integer>(initialVertexIdxs);
    ArrayList<Integer> bestEdges = memoizedBestJoins.get(allFactors);

    Query curQuery = QueryOptExperiment.getCurrentQuery();
    curQuery.joinOrders.put("LEFT_DEEP", new MyUtils.JoinOrder());
    ArrayList<int[]> joinOrder = new ArrayList<int[]>();

    for (Integer edge : bestEdges) {
        // FIXME: update the joinOrders
        int[] factorIndices = qg.edges.get(edge).factors.toArray();
        qg.updateGraph(factorIndices);
        joinOrder.add(factorIndices);
    }
    curQuery.joinOrders.get("LEFT_DEEP").joinEdgeChoices = joinOrder;
    RelNode optNode = qg.getFinalOptimalRelNode();
    call.transformTo(optNode);
  }

	/* Modified from:
	 * https://stackoverflow.com/questions/12548312/find-all-subsets-of-length-k-in-an-array
	 */
  private static void getSubsets(List<Integer> superSet, int k, int idx, Set<Integer> current,List<Set<Integer>> solution)
	{
    //successful stop clause
    if (current.size() == k) {
        solution.add(new HashSet<>(current));
        return;
    }
    //unsuccessful stop clause
    if (idx == superSet.size()) return;
    Integer x = superSet.get(idx);
    current.add(x);
    //"guess" x is in the subset
    getSubsets(superSet, k, idx+1, current, solution);
    current.remove(x);
    //"guess" x is not in the subset
    getSubsets(superSet, k, idx+1, current, solution);
	}
}

// End LeftDeepJoinOrderRule.java

