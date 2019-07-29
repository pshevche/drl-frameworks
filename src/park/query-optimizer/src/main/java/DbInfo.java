import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.rel.*;
import org.apache.calcite.rex.*;
import java.util.*;
import java.sql.*;
import org.apache.calcite.plan.RelOptUtil;

/* so we can get information about the whole database (including all tables)
 * while optimizing the queries.
 */
public class DbInfo {

  /* each table is mapped to its offset in the feature map */
  private static HashMap<String, Integer> tableFeaturesOffsetMap;
  private static ImmutableBitSet curVisibleFeatures = null;
  public static int attrCount;

  public static void init(CalciteConnection conn) throws SQLException {
    // Let's map each table to its offset in a one-hot feature encoding
    // scheme
    tableFeaturesOffsetMap = new HashMap<String, Integer>();
    DatabaseMetaData md = conn.getMetaData();
    String types[] = {"TABLE"};
    ResultSet tables = md.getTables(null, null, "%", types);
    //System.out.println("tables are: ");
    //int tableCount = 0;
    attrCount = 0;
    while (tables.next()) {
      String tableName = tables.getString(3);
      //System.out.println(tableName);
      tableFeaturesOffsetMap.put(tableName, attrCount);
      /* count all the attributes of this table */
      ResultSet attrs = md.getColumns(null, null, tableName, "%");
      while (attrs.next()) {
        attrCount += 1;
      }
    }
    System.out.println("final attr count: " + attrCount);
  }

  /* Need these are public getters so we can call them as
   * QueryOptExperiment.getAllTableFeaturesOffsets()
   */
  public static HashMap<String, Integer> getAllTableFeaturesOffsets() {
    return tableFeaturesOffsetMap;
  }

  /* @node: top level RelNode, i.e., it should have a projection either at
   * the top or so. (e.g., Agg ( Proj ( ...... )). This traverses through the
   * inputs of the projection to find the visible set of attributes for the
   * whole query.
   */
  private static ImmutableBitSet getDQFeatures(RelNode node) {
    ImmutableBitSet.Builder bld = ImmutableBitSet.builder();
    List<RexNode> projects = node.getChildExps();
    for (RexNode p : projects) {
      RelOptUtil.InputFinder inputFinder = new RelOptUtil.InputFinder();
      p.accept(inputFinder);
      ImmutableBitSet projectionBitSet = inputFinder.inputBitSet.build();
      bld.addAll(projectionBitSet);
    }

    for (RelNode n : node.getInputs()) {
      bld.addAll(getDQFeatures(n));
    }

    return bld.build();
  }

  public static void setCurrentQueryVisibleFeatures(RelNode node) {
    curVisibleFeatures = getDQFeatures(node);
  }

  // FIXME: will need to generalize this for parallel execution.
  public static ImmutableBitSet getCurrentQueryVisibleFeatures() {
    return curVisibleFeatures;
  }
}
