//TODO: reduce the number of imports
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinInfo;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.SemiJoin;
import org.apache.calcite.rel.metadata.RelColumnOrigin;
import org.apache.calcite.rel.metadata.RelMdUtil;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;

import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.BitSets;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.ImmutableIntList;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.mapping.IntPair;

import java.util.*;
import org.apache.calcite.rel.rules.*;

// experimental
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.plan.*;

public class JoinOrderTest extends RelOptRule {

  public static final JoinOrderTest INSTANCE =
      new JoinOrderTest(RelFactories.LOGICAL_BUILDER);

  public JoinOrderTest(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()), relBuilderFactory, null);
  }

  public JoinOrderTest(RelFactories.JoinFactory joinFactory,
      RelFactories.ProjectFactory projectFactory,
      RelFactories.FilterFactory filterFactory) {
    this(RelBuilder.proto(joinFactory, projectFactory, filterFactory));
  }

  private void printStuff(RelOptRuleCall call) {
    final MultiJoin multiJoinRel = call.rel(0);
    final LoptMultiJoin multiJoin = new LoptMultiJoin(multiJoinRel);
    final RelMetadataQuery mq = call.getMetadataQuery();
    final RexBuilder rexBuilder = multiJoinRel.getCluster().getRexBuilder();

    // can print out various stuff here. But still don't get the absolute
    // field names in the bigger tables.
    if (true) {
        List<RelNode> mjInputs = multiJoinRel.getInputs();
        for (RelNode rn : mjInputs) {
            System.out.println("class name : " + rn.getClass().getName());
            System.out.println("digest: " + rn.recomputeDigest());
            RelDataType dt = rn.getRowType();
            System.out.println("dt.toString: " + dt.toString());
            List<RelDataTypeField> dtFields = dt.getFieldList();
            for (RelDataTypeField rdtf : dtFields) {
                System.out.println("index: " + rdtf.getIndex());
                System.out.println("name: " + rdtf.getName());
            }
        }
        List<RexNode> mjConds = multiJoinRel.getOuterJoinConditions();
        for (RexNode rxn : mjConds) {
            System.out.println("join condition: " + rxn.toString());
        }
    }
  }

  public void onMatch(RelOptRuleCall call) {
      System.out.println("in my join rule's onMatch!");
      printStuff(call);
      System.exit(-1);
      final MultiJoin multiJoinRel = call.rel(0);

      final LoptMultiJoin multiJoin = new LoptMultiJoin(multiJoinRel);
      final RelMetadataQuery mq = call.getMetadataQuery();
      final RexBuilder rexBuilder = multiJoinRel.getCluster().getRexBuilder();
      List<RelNode> inputs = multiJoinRel.getInputs();

      ArrayList<Integer> randomOrder = new ArrayList<Integer>();
      for (int i = 0; i < multiJoinRel.getInputs().size(); i++) {
          randomOrder.add(i);
      }
      Collections.shuffle(randomOrder);
      System.out.println("random order is: " + randomOrder);
      LoptJoinTree joinTree = null;
      final List<String> fieldNames =
          multiJoin.getMultiJoinRel().getRowType().getFieldNames();
      final int nJoinFactors = multiJoin.getNumJoinFactors();
      final BitSet factorsToAdd = BitSets.range(0, nJoinFactors);
      final BitSet factorsAdded = new BitSet(nJoinFactors);
      final List<RexNode> filtersToAdd = new ArrayList<>(multiJoin.getJoinFilters());

      int prevFactor = -1;
      RelBuilder relBuilder = call.builder();
      System.out.println("before starting the randomOrder loop");
      for (int nextFactor : randomOrder) {
        System.out.println("nextFactor is: " + nextFactor);
        // FIXME:

        // if this is the first factor in the tree, create a join tree with
        // the single factor
        if (joinTree == null) {
            joinTree = new LoptJoinTree(inputs.get(nextFactor), nextFactor);
            System.out.println("generated a new joinTree");
            continue;
        }

        // how to use factorsNeeded?
        //BitSet factorsNeeded =
          //multiJoin.getFactorsRefByFactor(nextFactor).toBitSet();
        //if (multiJoin.isNullGenerating(nextFactor)) {
          //factorsNeeded.or(multiJoin.getOuterJoinFactors(nextFactor).toBitSet());
        //}
        //factorsNeeded.and(factorsAdded);

        final List<RexNode> tmpFilters = new ArrayList<>(filtersToAdd);

        JoinRelType joinType;
        if (multiJoin.getMultiJoinRel().isFullOuterJoin()) {
          assert multiJoin.getNumJoinFactors() == 2;
          joinType = JoinRelType.FULL;
        } else if (multiJoin.isNullGenerating(nextFactor)) {
          joinType = JoinRelType.LEFT;
        } else {
          joinType = JoinRelType.INNER;
        }
        System.out.println(joinType);

        LoptJoinTree rightTree = new LoptJoinTree(inputs.get(nextFactor),
            nextFactor);

        // in the case of a left or right outer join, use the specific
        // outer join condition
        RexNode condition;
        if ((joinType == JoinRelType.LEFT) || (joinType == JoinRelType.RIGHT)) {
          condition = multiJoin.getOuterJoinCond(nextFactor);
        } else {
          condition = addFilters( multiJoin, joinTree, -1, rightTree,
              filtersToAdd, false);
        }
        // FIXME: set last arg to value of selfJoin
        joinTree = createJoinSubtree(mq, relBuilder, multiJoin, joinTree,
            rightTree, condition, joinType, filtersToAdd, true, false);

          factorsToAdd.clear(nextFactor);
          factorsAdded.set(nextFactor);
          prevFactor = nextFactor;
      }

      RelNode newProject = createTopProject(call.builder(), multiJoin,
              joinTree, fieldNames);
      System.out.println("toString after random optimization: " +
			RelOptUtil.toString(newProject));

			call.getPlanner().setImportance(multiJoinRel, 0.00);
      call.transformTo(newProject);
  }

  /**
   * Creates the topmost projection that will sit on top of the selected join
   * ordering. The projection needs to match the original join ordering. Also,
   * places any post-join filters on top of the project.
   *
   * @param multiJoin join factors being optimized
   * @param joinTree selected join ordering
   * @param fieldNames field names corresponding to the projection expressions
   *
   * @return created projection
   */
  private RelNode createTopProject(
      RelBuilder relBuilder,
      LoptMultiJoin multiJoin,
      LoptJoinTree joinTree,
      List<String> fieldNames) {
    // shouldn't projections be the same?
    List<RexNode> newProjExprs = new ArrayList<>();
    RexBuilder rexBuilder =
        multiJoin.getMultiJoinRel().getCluster().getRexBuilder();

    // we don't need the damn tree order to get this.
    final List<Integer> newJoinOrder = joinTree.getTreeOrder();
    System.out.println("newJoinOrder is: " + newJoinOrder);

    int nJoinFactors = multiJoin.getNumJoinFactors();

    List<RelDataTypeField> fields = multiJoin.getMultiJoinFields();

    // create a mapping from each factor to its field offset in the join
    // ordering
    final Map<Integer, Integer> factorToOffsetMap = new HashMap<>();
    for (int pos = 0, fieldStart = 0; pos < nJoinFactors; pos++) {
      factorToOffsetMap.put(newJoinOrder.get(pos), fieldStart);
      fieldStart +=
          multiJoin.getNumFieldsInJoinFactor(newJoinOrder.get(pos));
    }

    // TODO: does this effect any of our samples?
    for (int currFactor = 0; currFactor < nJoinFactors; currFactor++) {
      // if the factor is the right factor in a removable self-join,
      // then where possible, remap references to the right factor to
      // the corresponding reference in the left factor
      Integer leftFactor = null;
      if (multiJoin.isRightFactorInRemovableSelfJoin(currFactor)) {
        leftFactor = multiJoin.getOtherSelfJoinFactor(currFactor);
      }
      for (int fieldPos = 0;
          fieldPos < multiJoin.getNumFieldsInJoinFactor(currFactor);
          fieldPos++) {
        int newOffset = factorToOffsetMap.get(currFactor) + fieldPos;
        if (leftFactor != null) {
          Integer leftOffset =
              multiJoin.getRightColumnMapping(currFactor, fieldPos);
          if (leftOffset != null) {
            newOffset =
                factorToOffsetMap.get(leftFactor) + leftOffset;
          }
        }
        newProjExprs.add(
            rexBuilder.makeInputRef(
                fields.get(newProjExprs.size()).getType(),
                newOffset));
      }
    }

    // TODO: what's happening here?
    relBuilder.push(joinTree.getJoinTree());
    relBuilder.project(newProjExprs, fieldNames);

    // Place the post-join filter (if it exists) on top of the final
    // projection.
    RexNode postJoinFilter =
        multiJoin.getMultiJoinRel().getPostJoinFilter();
    if (postJoinFilter != null) {
      relBuilder.filter(postJoinFilter);
    }
    return relBuilder.build();
  }

  /**
   * Returns whether a RelNode corresponds to a Join that wasn't one of the
   * original MultiJoin input factors.
   */
  private boolean isJoinTree(RelNode rel) {
    // full outer joins were already optimized in a prior instantiation
    // of this rule; therefore we should never see a join input that's
    // a full outer join
    if (rel instanceof Join) {
      assert ((Join) rel).getJoinType() != JoinRelType.FULL;
      return true;
    } else {
      return false;
    }
  }

  /**
   * Determines which join filters can be added to the current join tree. Note
   * that the join filter still reflects the original join ordering. It will
   * only be adjusted to reflect the new join ordering if the "adjust"
   * parameter is set to true.
   *
   * @param multiJoin join factors being optimized
   * @param leftTree left subtree of the join tree
   * @param leftIdx if &ge; 0, only consider filters that reference leftIdx in
   * leftTree; otherwise, consider all filters that reference any factor in
   * leftTree
   * @param rightTree right subtree of the join tree
   * @param filtersToAdd remaining join filters that need to be added; those
   * that are added are removed from the list
   * @param adjust if true, adjust filter to reflect new join ordering
   *
   * @return AND'd expression of the join filters that can be added to the
   * current join tree
   */
  private RexNode addFilters(
      LoptMultiJoin multiJoin,
      LoptJoinTree leftTree,
      int leftIdx,
      LoptJoinTree rightTree,
      List<RexNode> filtersToAdd,
      boolean adjust) {
    // loop through the remaining filters to be added and pick out the
    // ones that reference only the factors in the new join tree
    final RexBuilder rexBuilder =
        multiJoin.getMultiJoinRel().getCluster().getRexBuilder();
    final ImmutableBitSet.Builder childFactorBuilder =
        ImmutableBitSet.builder();
    childFactorBuilder.addAll(rightTree.getTreeOrder());
    if (leftIdx >= 0) {
      childFactorBuilder.set(leftIdx);
    } else {
      childFactorBuilder.addAll(leftTree.getTreeOrder());
    }
    for (int child : rightTree.getTreeOrder()) {
      childFactorBuilder.set(child);
    }

    final ImmutableBitSet childFactor = childFactorBuilder.build();
    RexNode condition = null;
    final ListIterator<RexNode> filterIter = filtersToAdd.listIterator();
    while (filterIter.hasNext()) {
      RexNode joinFilter = filterIter.next();
      ImmutableBitSet filterBitmap =
          multiJoin.getFactorsRefByJoinFilter(joinFilter);

      // if all factors in the join filter are in the join tree,
      // AND the filter to the current join condition
      if (childFactor.contains(filterBitmap)) {
        if (condition == null) {
          condition = joinFilter;
        } else {
          condition =
              rexBuilder.makeCall(
                  SqlStdOperatorTable.AND,
                  condition,
                  joinFilter);
        }
        filterIter.remove();
      }
    }

    if (adjust && (condition != null)) {
      int [] adjustments = new int[multiJoin.getNumTotalFields()];
      if (needsAdjustment(
          multiJoin,
          adjustments,
          leftTree,
          rightTree,
          false)) {
        condition =
            condition.accept(
                new RelOptUtil.RexInputConverter(
                    rexBuilder,
                    multiJoin.getMultiJoinFields(),
                    leftTree.getJoinTree().getRowType().getFieldList(),
                    rightTree.getJoinTree().getRowType().getFieldList(),
                    adjustments));
      }
    }

    if (condition == null) {
      condition = rexBuilder.makeLiteral(true);
    }

    return condition;
  }

  /**
   * Adjusts a filter to reflect a newly added factor in the middle of an
   * existing join tree
   *
   * @param multiJoin join factors being optimized
   * @param left left subtree of the join
   * @param right right subtree of the join
   * @param condition current join condition
   * @param factorAdded index corresponding to the newly added factor
   * @param origJoinOrder original join order, before factor was pushed into
   * the tree
   * @param origFields fields from the original join before the factor was
   * added
   *
   * @return modified join condition reflecting addition of the new factor
   */
  private RexNode adjustFilter(
      LoptMultiJoin multiJoin,
      LoptJoinTree left,
      LoptJoinTree right,
      RexNode condition,
      int factorAdded,
      List<Integer> origJoinOrder,
      List<RelDataTypeField> origFields) {
    final List<Integer> newJoinOrder = new ArrayList<>();
    left.getTreeOrder(newJoinOrder);
    right.getTreeOrder(newJoinOrder);

    int totalFields =
        left.getJoinTree().getRowType().getFieldCount()
            + right.getJoinTree().getRowType().getFieldCount()
            - multiJoin.getNumFieldsInJoinFactor(factorAdded);
    int [] adjustments = new int[totalFields];

    // go through each factor and adjust relative to the original
    // join order
    boolean needAdjust = false;
    int nFieldsNew = 0;
    for (int newPos = 0; newPos < newJoinOrder.size(); newPos++) {
      int nFieldsOld = 0;

      // no need to make any adjustments on the newly added factor
      int factor = newJoinOrder.get(newPos);
      if (factor != factorAdded) {
        // locate the position of the factor in the original join
        // ordering
        for (int pos : origJoinOrder) {
          if (factor == pos) {
            break;
          }
          nFieldsOld += multiJoin.getNumFieldsInJoinFactor(pos);
        }

        // fill in the adjustment array for this factor
        if (remapJoinReferences(
            multiJoin,
            factor,
            newJoinOrder,
            newPos,
            adjustments,
            nFieldsOld,
            nFieldsNew,
            false)) {
          needAdjust = true;
        }
      }
      nFieldsNew += multiJoin.getNumFieldsInJoinFactor(factor);
    }

    if (needAdjust) {
      RexBuilder rexBuilder =
          multiJoin.getMultiJoinRel().getCluster().getRexBuilder();
      condition =
          condition.accept(
              new RelOptUtil.RexInputConverter(
                  rexBuilder,
                  origFields,
                  left.getJoinTree().getRowType().getFieldList(),
                  right.getJoinTree().getRowType().getFieldList(),
                  adjustments));
    }

    return condition;
  }

  /**
   * Sets an adjustment array based on where column references for a
   * particular factor end up as a result of a new join ordering.
   *
   * <p>If the factor is not the right factor in a removable self-join, then
   * it needs to be adjusted as follows:
   *
   * <ul>
   * <li>First subtract, based on where the factor was in the original join
   * ordering.
   * <li>Then add on the number of fields in the factors that now precede this
   * factor in the new join ordering.
   * </ul>
   *
   * <p>If the factor is the right factor in a removable self-join and its
   * column reference can be mapped to the left factor in the self-join, then:
   *
   * <ul>
   * <li>First subtract, based on where the column reference is in the new
   * join ordering.
   * <li>Then, add on the number of fields up to the start of the left factor
   * in the self-join in the new join ordering.
   * <li>Then, finally add on the offset of the corresponding column from the
   * left factor.
   * </ul>
   *
   * <p>Note that this only applies if both factors in the self-join are in the
   * join ordering. If they are, then the left factor always precedes the
   * right factor in the join ordering.
   *
   * @param multiJoin join factors being optimized
   * @param factor the factor whose references are being adjusted
   * @param newJoinOrder the new join ordering containing the factor
   * @param newPos the position of the factor in the new join ordering
   * @param adjustments the adjustments array that will be set
   * @param offset the starting offset within the original join ordering for
   * the columns of the factor being adjusted
   * @param newOffset the new starting offset in the new join ordering for the
   * columns of the factor being adjusted
   * @param alwaysUseDefault always use the default adjustment value
   * regardless of whether the factor is the right factor in a removable
   * self-join
   *
   * @return true if at least one column from the factor requires adjustment
   */
  private boolean remapJoinReferences(
      LoptMultiJoin multiJoin,
      int factor,
      List<Integer> newJoinOrder,
      int newPos,
      int [] adjustments,
      int offset,
      int newOffset,
      boolean alwaysUseDefault) {
    boolean needAdjust = false;
    int defaultAdjustment = -offset + newOffset;
    if (!alwaysUseDefault
        && multiJoin.isRightFactorInRemovableSelfJoin(factor)
        && (newPos != 0)
        && newJoinOrder.get(newPos - 1).equals(
          multiJoin.getOtherSelfJoinFactor(factor))) {
      int nLeftFields =
          multiJoin.getNumFieldsInJoinFactor(
              newJoinOrder.get(
                  newPos - 1));
      for (int i = 0;
          i < multiJoin.getNumFieldsInJoinFactor(factor);
          i++) {
        Integer leftOffset = multiJoin.getRightColumnMapping(factor, i);

        // if the left factor doesn't reference the column, then
        // use the default adjustment value
        if (leftOffset == null) {
          adjustments[i + offset] = defaultAdjustment;
        } else {
          adjustments[i + offset] =
              -(offset + i) + (newOffset - nLeftFields)
                  + leftOffset;
        }
        if (adjustments[i + offset] != 0) {
          needAdjust = true;
        }
      }
    } else {
      if (defaultAdjustment != 0) {
        needAdjust = true;
        for (int i = 0;
            i < multiJoin.getNumFieldsInJoinFactor(
                newJoinOrder.get(newPos));
            i++) {
          adjustments[i + offset] = defaultAdjustment;
        }
      }
    }

    return needAdjust;
  }

  /**
   * In the event that a dimension table does not need to be joined because of
   * a semijoin, this method creates a join tree that consists of a projection
   * on top of an existing join tree. The existing join tree must contain the
   * fact table in the semijoin that allows the dimension table to be removed.
   *
   * <p>The projection created on top of the join tree mimics a join of the
   * fact and dimension tables. In order for the dimension table to have been
   * removed, the only fields referenced from the dimension table are its
   * dimension keys. Therefore, we can replace these dimension fields with the
   * fields corresponding to the semijoin keys from the fact table in the
   * projection.
   *
   * @param multiJoin join factors being optimized
   * @param semiJoinOpt optimal semijoins for each factor
   * @param factTree existing join tree containing the fact table
   * @param dimIdx dimension table factor id
   * @param filtersToAdd filters remaining to be added; filters added to the
   * new join tree are removed from the list
   *
   * @return created join tree or null if the corresponding fact table has not
   * been joined in yet
   */
  private LoptJoinTree createReplacementSemiJoin(
      RelBuilder relBuilder,
      LoptMultiJoin multiJoin,
      LoptSemiJoinOptimizer semiJoinOpt,
      LoptJoinTree factTree,
      int dimIdx,
      List<RexNode> filtersToAdd) {
    // if the current join tree doesn't contain the fact table, then
    // don't bother trying to create the replacement join just yet
    if (factTree == null) {
      return null;
    }

    int factIdx = multiJoin.getJoinRemovalFactor(dimIdx);
    final List<Integer> joinOrder = factTree.getTreeOrder();
    assert joinOrder.contains(factIdx);

    // figure out the position of the fact table in the current jointree
    int adjustment = 0;
    for (Integer factor : joinOrder) {
      if (factor == factIdx) {
        break;
      }
      adjustment += multiJoin.getNumFieldsInJoinFactor(factor);
    }

    // map the dimension keys to the corresponding keys from the fact
    // table, based on the fact table's position in the current jointree
    List<RelDataTypeField> dimFields =
        multiJoin.getJoinFactor(dimIdx).getRowType().getFieldList();
    int nDimFields = dimFields.size();
    Integer [] replacementKeys = new Integer[nDimFields];
    SemiJoin semiJoin = multiJoin.getJoinRemovalSemiJoin(dimIdx);
    ImmutableIntList dimKeys = semiJoin.getRightKeys();
    ImmutableIntList factKeys = semiJoin.getLeftKeys();
    for (int i = 0; i < dimKeys.size(); i++) {
      replacementKeys[dimKeys.get(i)] = factKeys.get(i) + adjustment;
    }

    return createReplacementJoin(
        relBuilder,
        multiJoin,
        semiJoinOpt,
        factTree,
        factIdx,
        dimIdx,
        dimKeys,
        replacementKeys,
        filtersToAdd);
  }

  /**
   * Creates a replacement join, projecting either dummy columns or
   * replacement keys from the factor that doesn't actually need to be joined.
   *
   * @param multiJoin join factors being optimized
   * @param semiJoinOpt optimal semijoins for each factor
   * @param currJoinTree current join tree being added to
   * @param leftIdx if &ge; 0, when creating the replacement join, only consider
   * filters that reference leftIdx in currJoinTree; otherwise, consider all
   * filters that reference any factor in currJoinTree
   * @param factorToAdd new factor whose join can be removed
   * @param newKeys join keys that need to be replaced
   * @param replacementKeys the keys that replace the join keys; null if we're
   * removing the null generating factor in an outer join
   * @param filtersToAdd filters remaining to be added; filters added to the
   * new join tree are removed from the list
   *
   * @return created join tree with an appropriate projection for the factor
   * that can be removed
   */
  private LoptJoinTree createReplacementJoin(
      RelBuilder relBuilder,
      LoptMultiJoin multiJoin,
      LoptSemiJoinOptimizer semiJoinOpt,
      LoptJoinTree currJoinTree,
      int leftIdx,
      int factorToAdd,
      ImmutableIntList newKeys,
      Integer [] replacementKeys,
      List<RexNode> filtersToAdd) {
    // create a projection, projecting the fields from the join tree
    // containing the current joinRel and the new factor; for fields
    // corresponding to join keys, replace them with the corresponding key
    // from the replacementKeys passed in; for other fields, just create a
    // null expression as a placeholder for the column; this is done so we
    // don't have to adjust the offsets of other expressions that reference
    // the new factor; the placeholder expression values should never be
    // referenced, so that's why it's ok to create these possibly invalid
    // expressions
    RelNode currJoinRel = currJoinTree.getJoinTree();
    List<RelDataTypeField> currFields = currJoinRel.getRowType().getFieldList();
    final int nCurrFields = currFields.size();
    List<RelDataTypeField> newFields =
        multiJoin.getJoinFactor(factorToAdd).getRowType().getFieldList();
    final int nNewFields = newFields.size();
    List<Pair<RexNode, String>> projects = new ArrayList<>();
    RexBuilder rexBuilder = currJoinRel.getCluster().getRexBuilder();
    RelDataTypeFactory typeFactory = rexBuilder.getTypeFactory();

    for (int i = 0; i < nCurrFields; i++) {
      projects.add(
          Pair.of(
              (RexNode) rexBuilder.makeInputRef(currFields.get(i).getType(), i),
              currFields.get(i).getName()));
    }
    for (int i = 0; i < nNewFields; i++) {
      RexNode projExpr;
      RelDataType newType = newFields.get(i).getType();
      if (!newKeys.contains(i)) {
        if (replacementKeys == null) {
          // null generating factor in an outer join; so make the
          // type nullable
          newType =
              typeFactory.createTypeWithNullability(newType, true);
        }
        projExpr =
            rexBuilder.makeCast(newType, rexBuilder.constantNull());
      } else {
        RelDataTypeField mappedField = currFields.get(replacementKeys[i]);
        RexNode mappedInput =
            rexBuilder.makeInputRef(
                mappedField.getType(),
                replacementKeys[i]);

        // if the types aren't the same, create a cast
        if (mappedField.getType() == newType) {
          projExpr = mappedInput;
        } else {
          projExpr =
              rexBuilder.makeCast(
                  newFields.get(i).getType(),
                  mappedInput);
        }
      }
      projects.add(Pair.of(projExpr, newFields.get(i).getName()));
    }

    relBuilder.push(currJoinRel);
    relBuilder.project(Pair.left(projects), Pair.right(projects));

    // remove the join conditions corresponding to the join we're removing;
    // we don't actually need to use them, but we need to remove them
    // from the list since they're no longer needed
    LoptJoinTree newTree =
        new LoptJoinTree(
            semiJoinOpt.getChosenSemiJoin(factorToAdd),
            factorToAdd);
    addFilters(
        multiJoin,
        currJoinTree,
        leftIdx,
        newTree,
        filtersToAdd,
        false);

    // Filters referencing factors other than leftIdx and factorToAdd
    // still do need to be applied.  So, add them into a separate
    // LogicalFilter placed on top off the projection created above.
    if (leftIdx >= 0) {
      addAdditionalFilters(
          relBuilder,
          multiJoin,
          currJoinTree,
          newTree,
          filtersToAdd);
    }

    // finally, create a join tree consisting of the current join's join
    // tree with the newly created projection; note that in the factor
    // tree, we act as if we're joining in the new factor, even
    // though we really aren't; this is needed so we can map the columns
    // from the new factor as we go up in the join tree
    return new LoptJoinTree(
        relBuilder.build(),
        currJoinTree.getFactorTree(),
        newTree.getFactorTree());
  }

  /**
   * Creates a LogicalJoin given left and right operands and a join condition.
   * Swaps the operands if beneficial.
   *
   * @param multiJoin join factors being optimized
   * @param left left operand
   * @param right right operand
   * @param condition join condition
   * @param joinType the join type
   * @param fullAdjust true if the join condition reflects the original join
   * ordering and therefore has not gone through any type of adjustment yet;
   * otherwise, the condition has already been partially adjusted and only
   * needs to be further adjusted if swapping is done
   * @param filtersToAdd additional filters that may be added on top of the
   * resulting LogicalJoin, if the join is a left or right outer join
   * @param selfJoin true if the join being created is a self-join that's
   * removable
   *
   * @return created LogicalJoin
   */
  private LoptJoinTree createJoinSubtree(
      RelMetadataQuery mq,
      RelBuilder relBuilder,
      LoptMultiJoin multiJoin,
      LoptJoinTree left,
      LoptJoinTree right,
      RexNode condition,
      JoinRelType joinType,
      List<RexNode> filtersToAdd,
      boolean fullAdjust,
      boolean selfJoin) {
    RexBuilder rexBuilder =
        multiJoin.getMultiJoinRel().getCluster().getRexBuilder();

    // swap the inputs if beneficial
    if (swapInputs(mq, multiJoin, left, right, selfJoin)) {
      LoptJoinTree tmp = right;
      right = left;
      left = tmp;
      if (!fullAdjust) {
        condition =
            swapFilter(
                rexBuilder,
                multiJoin,
                right,
                left,
                condition);
      }
      if ((joinType != JoinRelType.INNER)
          && (joinType != JoinRelType.FULL)) {
        joinType =
            (joinType == JoinRelType.LEFT) ? JoinRelType.RIGHT
                : JoinRelType.LEFT;
      }
    }

    if (fullAdjust) {
      int [] adjustments = new int[multiJoin.getNumTotalFields()];
      if (needsAdjustment(
          multiJoin,
          adjustments,
          left,
          right,
          selfJoin)) {
        condition =
            condition.accept(
                new RelOptUtil.RexInputConverter(
                    rexBuilder,
                    multiJoin.getMultiJoinFields(),
                    left.getJoinTree().getRowType().getFieldList(),
                    right.getJoinTree().getRowType().getFieldList(),
                    adjustments));
      }
    }

    relBuilder.push(left.getJoinTree())
        .push(right.getJoinTree())
        .join(joinType, condition);

    // if this is a left or right outer join, and additional filters can
    // be applied to the resulting join, then they need to be applied
    // as a filter on top of the outer join result
    if ((joinType == JoinRelType.LEFT) || (joinType == JoinRelType.RIGHT)) {
      assert !selfJoin;
      addAdditionalFilters(
          relBuilder,
          multiJoin,
          left,
          right,
          filtersToAdd);
    }

    return new LoptJoinTree(
        relBuilder.build(),
        left.getFactorTree(),
        right.getFactorTree(),
        selfJoin);
  }

  /**
   * Determines whether any additional filters are applicable to a join tree.
   * If there are any, creates a filter node on top of the join tree with the
   * additional filters.
   *
   * @param relBuilder Builder holding current join tree
   * @param multiJoin join factors being optimized
   * @param left left side of join tree
   * @param right right side of join tree
   * @param filtersToAdd remaining filters
   */
  private void addAdditionalFilters(
      RelBuilder relBuilder,
      LoptMultiJoin multiJoin,
      LoptJoinTree left,
      LoptJoinTree right,
      List<RexNode> filtersToAdd) {
    RexNode filterCond =
        addFilters(multiJoin, left, -1, right, filtersToAdd, false);
    if (!filterCond.isAlwaysTrue()) {
      // adjust the filter to reflect the outer join output
      int [] adjustments = new int[multiJoin.getNumTotalFields()];
      if (needsAdjustment(multiJoin, adjustments, left, right, false)) {
        RexBuilder rexBuilder =
            multiJoin.getMultiJoinRel().getCluster().getRexBuilder();
        filterCond =
            filterCond.accept(
                new RelOptUtil.RexInputConverter(
                    rexBuilder,
                    multiJoin.getMultiJoinFields(),
                    relBuilder.peek().getRowType().getFieldList(),
                    adjustments));
        relBuilder.filter(filterCond);
      }
    }
  }

  /**
   * Swaps the operands to a join, so the smaller input is on the right. Or,
   * if this is a removable self-join, swap so the factor that should be
   * preserved when the self-join is removed is put on the left.
   *
   * @param multiJoin join factors being optimized
   * @param left left side of join tree
   * @param right right hand side of join tree
   * @param selfJoin true if the join is a removable self-join
   *
   * @return true if swapping should be done
   */
  private boolean swapInputs(
      RelMetadataQuery mq,
      LoptMultiJoin multiJoin,
      LoptJoinTree left,
      LoptJoinTree right,
      boolean selfJoin) {
    boolean swap = false;

    //if (selfJoin) {
      //return !multiJoin.isLeftFactorInRemovableSelfJoin(
          //((LoptJoinTree.Leaf) left.getFactorTree()).getId());
    //}

    final Double leftRowCount = mq.getRowCount(left.getJoinTree());
    final Double rightRowCount = mq.getRowCount(right.getJoinTree());

    // The left side is smaller than the right if it has fewer rows,
    // or if it has the same number of rows as the right (excluding
    // roundoff), but fewer columns.
    if ((leftRowCount != null)
        && (rightRowCount != null)
        && ((leftRowCount < rightRowCount)
        || ((Math.abs(leftRowCount - rightRowCount)
        < RelOptUtil.EPSILON)
        && (rowWidthCost(left.getJoinTree())
        < rowWidthCost(right.getJoinTree()))))) {
      swap = true;
    }
    return swap;
  }

  /**
   * Adjusts a filter to reflect swapping of join inputs
   *
   * @param rexBuilder rexBuilder
   * @param multiJoin join factors being optimized
   * @param origLeft original LHS of the join tree (before swap)
   * @param origRight original RHS of the join tree (before swap)
   * @param condition original join condition
   *
   * @return join condition reflect swap of join inputs
   */
  private RexNode swapFilter(
      RexBuilder rexBuilder,
      LoptMultiJoin multiJoin,
      LoptJoinTree origLeft,
      LoptJoinTree origRight,
      RexNode condition) {
    int nFieldsOnLeft =
        origLeft.getJoinTree().getRowType().getFieldCount();
    int nFieldsOnRight =
        origRight.getJoinTree().getRowType().getFieldCount();
    int [] adjustments = new int[nFieldsOnLeft + nFieldsOnRight];

    for (int i = 0; i < nFieldsOnLeft; i++) {
      adjustments[i] = nFieldsOnRight;
    }
    for (int i = nFieldsOnLeft; i < (nFieldsOnLeft + nFieldsOnRight); i++) {
      adjustments[i] = -nFieldsOnLeft;
    }

    condition =
        condition.accept(
            new RelOptUtil.RexInputConverter(
                rexBuilder,
                multiJoin.getJoinFields(origLeft, origRight),
                multiJoin.getJoinFields(origRight, origLeft),
                adjustments));

    return condition;
  }

  /**
   * Sets an array indicating how much each factor in a join tree needs to be
   * adjusted to reflect the tree's join ordering
   *
   * @param multiJoin join factors being optimized
   * @param adjustments array to be filled out
   * @param joinTree join tree
   * @param otherTree null unless joinTree only represents the left side of
   * the join tree
   * @param selfJoin true if no adjustments need to be made for self-joins
   *
   * @return true if some adjustment is required; false otherwise
   */
  private boolean needsAdjustment(
      LoptMultiJoin multiJoin,
      int [] adjustments,
      LoptJoinTree joinTree,
      LoptJoinTree otherTree,
      boolean selfJoin) {
    boolean needAdjustment = false;

    final List<Integer> joinOrder = new ArrayList<>();
    joinTree.getTreeOrder(joinOrder);
    if (otherTree != null) {
      otherTree.getTreeOrder(joinOrder);
    }

    int nFields = 0;
    for (int newPos = 0; newPos < joinOrder.size(); newPos++) {
      int origPos = joinOrder.get(newPos);
      int joinStart = multiJoin.getJoinStart(origPos);

      // Determine the adjustments needed for join references.  Note
      // that if the adjustment is being done for a self-join filter,
      // we always use the default adjustment value rather than
      // remapping the right factor to reference the left factor.
      // Otherwise, we have no way of later identifying that the join is
      // self-join.
      if (remapJoinReferences(
          multiJoin,
          origPos,
          joinOrder,
          newPos,
          adjustments,
          joinStart,
          nFields,
          selfJoin)) {
        needAdjustment = true;
      }
      nFields += multiJoin.getNumFieldsInJoinFactor(origPos);
    }

    return needAdjustment;
  }

  /**
   * Determines whether a join is a removable self-join. It is if it's an
   * inner join between identical, simple factors and the equality portion of
   * the join condition consists of the same set of unique keys.
   *
   * @param joinRel the join
   *
   * @return true if the join is removable
   */
  public static boolean isRemovableSelfJoin(Join joinRel) {
    final RelNode left = joinRel.getLeft();
    final RelNode right = joinRel.getRight();

    if (joinRel.getJoinType() != JoinRelType.INNER) {
      return false;
    }

    // Make sure the join is between the same simple factor
    final RelMetadataQuery mq = joinRel.getCluster().getMetadataQuery();
    final RelOptTable leftTable = mq.getTableOrigin(left);
    if (leftTable == null) {
      return false;
    }
    final RelOptTable rightTable = mq.getTableOrigin(right);
    if (rightTable == null) {
      return false;
    }
    if (!leftTable.getQualifiedName().equals(rightTable.getQualifiedName())) {
      return false;
    }

    // Determine if the join keys are identical and unique
    return areSelfJoinKeysUnique(mq, left, right, joinRel.getCondition());
  }

  /**
   * Determines if the equality portion of a self-join condition is between
   * identical keys that are unique.
   *
   * @param mq Metadata query
   * @param leftRel left side of the join
   * @param rightRel right side of the join
   * @param joinFilters the join condition
   *
   * @return true if the equality join keys are the same and unique
   */
  private static boolean areSelfJoinKeysUnique(RelMetadataQuery mq,
      RelNode leftRel, RelNode rightRel, RexNode joinFilters) {
    final JoinInfo joinInfo = JoinInfo.of(leftRel, rightRel, joinFilters);

    // Make sure each key on the left maps to the same simple column as the
    // corresponding key on the right
    for (IntPair pair : joinInfo.pairs()) {
      final RelColumnOrigin leftOrigin =
          mq.getColumnOrigin(leftRel, pair.source);
      if (leftOrigin == null) {
        return false;
      }
      final RelColumnOrigin rightOrigin =
          mq.getColumnOrigin(rightRel, pair.target);
      if (rightOrigin == null) {
        return false;
      }
      if (leftOrigin.getOriginColumnOrdinal()
          != rightOrigin.getOriginColumnOrdinal()) {
        return false;
      }
    }

    // Now that we've verified that the keys are the same, see if they
    // are unique.  When removing self-joins, if needed, we'll later add an
    // IS NOT NULL filter on the join keys that are nullable.  Therefore,
    // it's ok if there are nulls in the unique key.
    return RelMdUtil.areColumnsDefinitelyUniqueWhenNullsFiltered(mq, leftRel,
        joinInfo.leftSet());
  }

   /**
   * Computes a cost for a join tree based on the row widths of the inputs
   * into the join. Joins where the inputs have the fewest number of columns
   * lower in the tree are better than equivalent joins where the inputs with
   * the larger number of columns are lower in the tree.
   *
   * @param tree a tree of RelNodes
   *
   * @return the cost associated with the width of the tree
   */
  private int rowWidthCost(RelNode tree) {
    // The width cost is the width of the tree itself plus the widths
    // of its children.  Hence, skinnier rows are better when they're
    // lower in the tree since the width of a RelNode contributes to
    // the cost of each LogicalJoin that appears above that RelNode.
    int width = tree.getRowType().getFieldCount();
    if (isJoinTree(tree)) {
      Join joinRel = (Join) tree;
      width +=
          rowWidthCost(joinRel.getLeft())
              + rowWidthCost(joinRel.getRight());
    }
    return width;
  }
}
