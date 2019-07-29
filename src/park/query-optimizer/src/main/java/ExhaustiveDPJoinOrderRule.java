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
public class ExhaustiveDPJoinOrderRule extends RelOptRule
{
  public static final ExhaustiveDPJoinOrderRule INSTANCE =
      new ExhaustiveDPJoinOrderRule(RelFactories.LOGICAL_BUILDER);

  private class IntermediateJoinState {
    ArrayList<ImmutableBitSet[]> bestJoins;
    double cost;
    public IntermediateJoinState(ArrayList<ImmutableBitSet[]> bestJoins, double
        cost)
    {
      this.bestJoins = bestJoins;
      this.cost = cost;
    }
  }
  // The keys represent a set of factors in the original vertices of the
  // QueryGraph. The values represent a sequence of edges (it's index at each
  // stage in QueryGraph.edges) that are chosen for the optimal memoized
  // ordering for the given set of factors. We can use these to reconstruct the
  // QueryGraph with a sequence of updateGraph steps for each
  // of the edge
  //private HashMap<ImmutableBitSet, ArrayList<ImmutableBitSet[]>> memoizedBestJoins;
  private HashMap<ImmutableBitSet, IntermediateJoinState> memoizedBestJoins;

  /** Creates an ExhaustiveDPJoinOrderRule. */
  public ExhaustiveDPJoinOrderRule(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()), relBuilderFactory, null);
  }

  @Deprecated // to be removed before 2.0
  public ExhaustiveDPJoinOrderRule(RelFactories.JoinFactory joinFactory,
      RelFactories.ProjectFactory projectFactory) {
    this(RelBuilder.proto(joinFactory, projectFactory));
  }

	/*
	 * We follow the algorithm described at:
   * https://dsg.uwaterloo.ca/seminars/notes/Guido.pdf
	 */
  @Override
  public void onMatch(RelOptRuleCall call)
  {
    RelNode orig = call.getRelList().get(0);
    call.getPlanner().setImportance(orig, 0.0);
    //memoizedBestJoins = new HashMap<ImmutableBitSet, ArrayList<ImmutableBitSet[]>>();
    memoizedBestJoins = new HashMap<ImmutableBitSet, IntermediateJoinState>();
    final MultiJoin multiJoinRel = call.rel(0);
    final RexBuilder rexBuilder = multiJoinRel.getCluster().getRexBuilder();
    final MyMetadataQuery mq = MyMetadataQuery.instance();
    final LoptMultiJoin multiJoin = new LoptMultiJoin(multiJoinRel);
    for (int i = 0; i < multiJoin.getNumJoinFactors(); i++) {
			 // no edges have been chosen yet, so we add an empty list
			 memoizedBestJoins.put(ImmutableBitSet.range(i, i+1),
           new IntermediateJoinState(new ArrayList<ImmutableBitSet[]>(), 0.00));
    }
    QueryGraph startQG = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());
    Iterator<ImmutableBitSet[]> csgCmpIt = startQG.csgCmpIterator();
    while (csgCmpIt.hasNext()) {
      ImmutableBitSet[] curPair = csgCmpIt.next();
      ImmutableBitSet S1 = curPair[0];
      ImmutableBitSet S2 = curPair[1];
      ImmutableBitSet S = S1.union(S2);
      ArrayList<ImmutableBitSet[]> p1 = memoizedBestJoins.get(S1).bestJoins;
      ArrayList<ImmutableBitSet[]> p2 = memoizedBestJoins.get(S2).bestJoins;
      assert p1 != null;
      assert p2 != null;
      QueryGraph curQG = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());

      // NOTE: S1, and S2, are two subgraphs with no common elements. So we can
      // execute the optimal ordering for those (p1, and p2) in either order,
      // since the factors are completely independent, thus joining factors in
      // one subgraph will NOT affect the indices of the factors in the other
      // subgraph.
      for (ImmutableBitSet[] factors : p1) {
        //System.out.println("p1 factors[0]: " + factors[0]);
        //System.out.println("p1 factors[1]: " + factors[1]);
        curQG.updateGraphBitset(factors);
      }
      int factor1Idx = curQG.allVertexes.size()-1;
      for (ImmutableBitSet[] factors : p2) {
        //System.out.println("p2 factors[0]: " + factors[0]);
        //System.out.println("p2 factors[1]: " + factors[1]);
        curQG.updateGraphBitset(factors);
      }
      for (QueryGraph.Vertex v : curQG.allVertexes) {
        if (v != null) {
          //System.out.println("v.factors: " + v.factors);
        }
      }
      // last vertex added must be the one because of factors in p2.
      int factor2Idx = curQG.allVertexes.size()-1;
      assert factor1Idx != factor2Idx;
      // now, cost of joining the two latest nodes.
      ImmutableBitSet[] lastFactors = {S1, S2};
      //System.out.println("lastFactors1: " + lastFactors[0]);
      //System.out.println("lastFactors2: " + lastFactors[1]);
      curQG.updateGraphBitset(lastFactors);
      double curCost = curQG.costSoFar;
      IntermediateJoinState bestOrder = memoizedBestJoins.get(S);
      ArrayList<ImmutableBitSet[]> curOrder = new ArrayList<ImmutableBitSet[]>();
      curOrder.addAll(p1);
      curOrder.addAll(p2);
      curOrder.add(lastFactors);
      if (bestOrder == null) {
        // first time we see the subgraph S
        memoizedBestJoins.put(S, new IntermediateJoinState(curOrder, curCost));
      } else {
        // find the cost of bestOrder, and replace if it needed.
        if (bestOrder.cost > curCost) {
          memoizedBestJoins.put(S, new IntermediateJoinState(curOrder, curCost));
        }
      }
    }

    // Not checking for null here, as we MUST have this in memoizedBestJoins,
    // or else something is wrong with the algorithm and it might as well
    // crash.
    ArrayList<ImmutableBitSet[]> optOrdering =
      memoizedBestJoins.get(ImmutableBitSet.range(0,
            multiJoin.getNumJoinFactors())).bestJoins;
    QueryGraph finalQG = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());
    Query curQuery = QueryOptExperiment.getCurrentQuery();
    //curQuery.joinOrders.put("EXHAUSTIVE", new ArrayList<int[]>());
    curQuery.joinOrders.put("EXHAUSTIVE", new MyUtils.JoinOrder());
    ArrayList<int[]> joinOrder = new ArrayList<int[]>();

    for (ImmutableBitSet[] factors : optOrdering) {
      int[] factorIndices = finalQG.updateGraphBitset(factors);
      joinOrder.add(factorIndices);
    }
    curQuery.joinOrders.get("EXHAUSTIVE").joinEdgeChoices = joinOrder;
    RelNode optNode = finalQG.getFinalOptimalRelNode();
    call.transformTo(optNode);
  }
}

// End ExhaustiveDPJoinOrderRule.java

