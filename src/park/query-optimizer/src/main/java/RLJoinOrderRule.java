/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
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
 * TODO: describe bushy rule which served as the base template etc.
 *
 * <p>It is triggered by the pattern
 * {@link org.apache.calcite.rel.logical.LogicalProject} ({@link MultiJoin}).
 *
 * <p>It is similar to
 * {@link org.apache.calcite.rel.rules.LoptOptimizeJoinRule}.
 * {@code LoptOptimizeJoinRule}
 */
public class RLJoinOrderRule extends RelOptRule {
  public static final RLJoinOrderRule INSTANCE =
      new RLJoinOrderRule(RelFactories.LOGICAL_BUILDER);

  private final PrintWriter pw = null;

  /** Creates an RLJoinOrderRule. */
  public RLJoinOrderRule(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()), relBuilderFactory, null);
  }

  @Deprecated // to be removed before 2.0
  public RLJoinOrderRule(RelFactories.JoinFactory joinFactory,
      RelFactories.ProjectFactory projectFactory) {
    this(RelBuilder.proto(joinFactory, projectFactory));
  }

  @Override
  public void onMatch(RelOptRuleCall call)
  {
    // setting original expression's importance to 0
    RelNode orig = call.getRelList().get(0);
    call.getPlanner().setImportance(orig, 0.0);

    ZeroMQServer zmq = QueryOptExperiment.getZMQServer();
    // this is required if we want to use node.computeSelfCost()
    // final RelOptPlanner planner = call.getPlanner();
    final MultiJoin multiJoinRel = call.rel(0);
    final RexBuilder rexBuilder = multiJoinRel.getCluster().getRexBuilder();
    final RelBuilder relBuilder = call.builder();
    QueryOptExperiment.Params params = QueryOptExperiment.getParams();

    // wrapper around RelMetadataQuery, to add support for non linear cost
    // models.
    final MyMetadataQuery mq = MyMetadataQuery.instance();
    //final MyMetadataQuery mq = QueryOptExperiment.getMetadataQuery();
    final LoptMultiJoin multiJoin = new LoptMultiJoin(multiJoinRel);

    QueryGraph queryGraph = new QueryGraph(multiJoin, mq, rexBuilder, relBuilder);
    zmq.queryGraph = queryGraph;

    // only used for finalReward scenario
    Double costSoFar = 0.00;
    zmq.episodeDone = 0;
    Query curQuery = QueryOptExperiment.getCurrentQuery();
    // replace whatever was there before
    //curQuery.joinOrders.put("RL", new ArrayList<int[]>());
    curQuery.joinOrders.put("RL", new MyUtils.JoinOrder());
    ArrayList<int[]> joinOrder = new ArrayList<int[]>();

    for (;;) {
      // break condition
      final int[] factors = chooseNextEdge(queryGraph);
      if (factors == null) break;
      joinOrder.add(factors);
      double cost = queryGraph.updateGraph(factors);
      if (queryGraph.edges.size() == 0) zmq.episodeDone = 1;

      // FIXME: onlyFinalReward is pretty hacky right now...
      costSoFar += cost;    // only required for onlyFinalReward
      zmq.lastReward = -cost;
      zmq.waitForClientTill("getReward");
    }
    curQuery.joinOrders.get("RL").joinEdgeChoices = joinOrder;

    /// FIXME: need to understand what this TargetMapping business really is...
    /// just adding a projection on top of the left nodes we had.
    final Pair<RelNode, Mappings.TargetMapping> top = Util.last(queryGraph.relNodes);
    relBuilder.push(top.left)
        .project(relBuilder.fields(top.right));
    RelNode optNode = relBuilder.build();
    call.transformTo(optNode);
  }

  /*
   * Passes control to the python agent to choose the next edge.
   * @ret: factors associated with the chosen edge
   */
  private int [] chooseNextEdge(QueryGraph queryGraph)
  {
    if (queryGraph.edges.size() == 0) {
      final QueryGraph.Vertex lastVertex = Util.last(queryGraph.allVertexes);
      final int z = lastVertex.factors.previousClearBit(lastVertex.id - 1);
      if (z < 0) {
        return null;
      }
      return new int[] {z, lastVertex.id};
    }

    final int[] factors;
    ZeroMQServer zmq = QueryOptExperiment.getZMQServer();
    // each edge is equivalent to a possible action, and must be represented
    // by its features
    final int edgeOrdinal;
    zmq.waitForClientTill("step");
    if (zmq.reset) {
      // TODO: put this option in QueryGraph.
      edgeOrdinal = ThreadLocalRandom.current().nextInt(0, queryGraph.edges.size());
    } else {
      edgeOrdinal = zmq.nextAction;
    }
    final QueryGraph.Edge bestEdge = queryGraph.edges.get(edgeOrdinal);

    // For now, assume that the edge is between precisely two factors.
    // 1-factor conditions have probably been pushed down,
    // and 3-or-more-factor conditions are advanced.
    assert bestEdge.factors.cardinality() == 2;
    factors = bestEdge.factors.toArray();
    return factors;
  }
}

// End RLJoinOrderRule.java

