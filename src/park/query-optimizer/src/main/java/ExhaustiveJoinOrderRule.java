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
 */
public class ExhaustiveJoinOrderRule extends RelOptRule {
  public static final ExhaustiveJoinOrderRule INSTANCE =
      new ExhaustiveJoinOrderRule(RelFactories.LOGICAL_BUILDER);

  // TODO: just set this to what we want.
  //private final PrintWriter pw = Util.printWriter(System.out);
  private final PrintWriter pw = null;
  private MyMetadataQuery mq;
  private int numAdded = 0;
  // just some large positive number to initialize, costs always positive, and
  // lower is better.
  private Double bestCost = Double.POSITIVE_INFINITY;
  private RelNode bestOptNode = null;
  private int totalOpts;
  private LoptMultiJoin multiJoin = null;
  private RexBuilder rexBuilder = null;

  /** Creates an ExhaustiveJoinOrderRule. */
  public ExhaustiveJoinOrderRule(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()), relBuilderFactory, null);
  }

  @Deprecated // to be removed before 2.0
  public ExhaustiveJoinOrderRule(RelFactories.JoinFactory joinFactory,
      RelFactories.ProjectFactory projectFactory) {
    this(RelBuilder.proto(joinFactory, projectFactory));
  }

  @Override
  public void onMatch(RelOptRuleCall call)
  {
    bestCost = Double.POSITIVE_INFINITY;
    bestOptNode = null;
    totalOpts = 0;
    // Setting original expressions importance to 0, so our choice will be
    // chosen.
    RelNode orig = call.getRelList().get(0);
    call.getPlanner().setImportance(orig, 0.0);
    final MultiJoin multiJoinRel = call.rel(0);
    rexBuilder = multiJoinRel.getCluster().getRexBuilder();
    //final RelBuilder relBuilder = call.builder();
    multiJoin = new LoptMultiJoin(multiJoinRel);
    mq = MyMetadataQuery.instance();
    QueryGraph qg = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());

    System.out.println("total edges: " + qg.edges.size());
    // FIXME: temporary
    if (qg.edges.size() >= 12) {
      return;
    }
    numAdded = 0;
    recursiveAddNodes(call, new ArrayList<Integer>(), qg);
    assert bestOptNode != null;
    //assert bestVertexes != null;
    // build a final optimized node using the bestVertexes
    //System.out.println("bestCost: " + bestCost);
    System.out.println("exhaustive search optNode cost " + mq.getCumulativeCost(bestOptNode));
    call.transformTo(bestOptNode);
  }

  private void recursiveAddNodes(RelOptRuleCall call, ArrayList<Integer> usedEdges, QueryGraph curQG)
  {
    totalOpts += 1;
    if (totalOpts % 1000 == 0) {
      System.out.println("totalOpts = " + totalOpts);
    }
    if (curQG.edges.size() == 0) {
      // do stuff
      //QueryGraph qg = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());
      //for (Integer edge: usedEdges) {
        //qg.updateGraph(qg.edges.get(edge).factors.toArray());
      //}
      //RelNode optNode = curQG.getFinalOptimalRelNode();
      //MyCost newCost = (MyCost) ((MyMetadataQuery) mq).getCumulativeCost(optNode);
      //Double finalCost = newCost.getCost();
      //if (finalCost < bestCost) {
      if (curQG.costSoFar < bestCost) {
        bestOptNode = curQG.getFinalOptimalRelNode();
        bestCost = curQG.costSoFar;
        //bestCost = finalCost;
      }
    }

    for (int i = 0; i < curQG.edges.size(); i++) {
      // try to use this edge next
      ArrayList<Integer> newUsedEdges = new ArrayList<Integer>(usedEdges);
      // how costly would choosing unusedEdge be?
      newUsedEdges.add(i);
      QueryGraph newQG = new QueryGraph(multiJoin, mq, rexBuilder, call.builder());
      for (Integer edge: newUsedEdges) {
        newQG.updateGraph(newQG.edges.get(edge).factors.toArray());
      }
      if (newQG.costSoFar >= bestCost) {
        continue;
      }
      recursiveAddNodes(call, newUsedEdges, newQG);
    }
  }
}

// End ExhaustiveJoinOrderRule.java
