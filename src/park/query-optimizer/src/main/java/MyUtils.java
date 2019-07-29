import java.util.*;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.plan.volcano.*;
import org.apache.calcite.plan.hep.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.type.*;

// FIXME: copied over from QueryOptExp
import java.sql.*;
import java.util.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.tools.*;
import java.io.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.*;
import org.apache.calcite.adapter.enumerable.*;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.plan.hep.*;
import org.apache.calcite.adapter.enumerable.*;

import com.google.common.collect.ImmutableList;
import org.apache.calcite.config.Lex;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.adapter.jdbc.JdbcSchema;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.commons.io.FileUtils;
import java.util.concurrent.ThreadLocalRandom;

// experimental:
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.util.ImmutableBitSet;
import java.util.concurrent.TimeUnit;
import org.apache.commons.io.IOUtils;
import java.nio.charset.StandardCharsets;

public class MyUtils {

  public static class JoinOrder {
    ArrayList<ArrayList<ArrayList<String>>> joinEdges = null;
    String joinStr = "";
    int joinCount = 0;
    // represents each of the join edge choice made based on the QueryGraph
    // representation used in each of the join-order search rules
    List<int[]> joinEdgeChoices = null;
  }

  public static MyUtils.JoinOrder updateJoinOrder(RelNode rootRel, JoinOrder jo) {
    /** Visitor that counts join nodes. */
    class JoinOrderFinder extends RelVisitor {
      ArrayList<ArrayList<ArrayList<String>>> joinEdges = new ArrayList<ArrayList<ArrayList<String>>>();
      String joinStr = "";
      int joinCount = 0;

      private String getJoinOrder(RelNode node) {
        String str = "";
        if (node instanceof Join) {
          ++joinCount;
          Join j = (Join) node;
          RelNode leftNode = j.getLeft();
          RelNode rightNode = j.getRight();
          ArrayList<String> leftTables = getAllTableNames(leftNode);
          ArrayList<String> rightTables = getAllTableNames(rightNode);
          ArrayList<String> curTables = getAllTableNames(node);
          java.util.Collections.sort(leftTables);
          java.util.Collections.sort(rightTables);
          java.util.Collections.sort(curTables);
          ArrayList<ArrayList<String>> edge1 = new ArrayList<ArrayList<String>>();
          ArrayList<ArrayList<String>> edge2 = new ArrayList<ArrayList<String>>();
          edge1.add(leftTables);
          edge1.add(curTables);
          edge2.add(rightTables);
          edge2.add(curTables);
          joinEdges.add(edge1);
          joinEdges.add(edge2);
          String left = getJoinOrder(leftNode);
          String right = getJoinOrder(rightNode);
          str = "JOIN (" + left + " , " + right + ")";
          return str;
        } else if (node instanceof Filter || node instanceof TableScan) {
          str = getTableName(node);
          return str;
        }
        return null;
      }

      @Override public void visit(RelNode node, int ordinal, RelNode parent) {
        if (node instanceof Join) {
          joinStr = getJoinOrder(node);
          return;
        } else {
          super.visit(node, ordinal, parent);
        }
      }

      JoinOrder run(RelNode node, JoinOrder jo) {
        go(node);
        if (jo == null) {
          jo = new JoinOrder();
        }
        jo.joinStr = joinStr;
        jo.joinCount = joinCount;
        jo.joinEdges = joinEdges;
        return jo;
      }
    }

    return new JoinOrderFinder().run(rootRel, jo);
  }

  public static ArrayList<String> getAllTableNames(RelNode rel) {
    if (rel == null) {
      return null;
    }
    List<RelNode> inputs = rel.getInputs();
    ArrayList<String> tableNames = new ArrayList<String>();
    if (inputs.size() <= 1) {
      String curTable = getTableName(rel);
      tableNames.add(curTable);
    } else {
      for (RelNode inp : inputs) {
        ArrayList<String> curTables = getAllTableNames(inp);
        if (curTables.size() >= 1) {
          tableNames.addAll(curTables);
        }
      }
    }
    return tableNames;
  }

  public static String getTableName(RelNode rel) {
    if (rel == null) {
      return null;
    }
    if (rel instanceof RelSubset) {
      RelSubset s = (RelSubset) rel;
      return getTableName(s.getOriginal());
    } else if (rel instanceof Filter) {
      return getTableName(rel.getInput(0));
    } else if (rel instanceof HepRelVertex) {
      return getTableName(((HepRelVertex) rel).getCurrentRel());
    } else if (rel instanceof TableScan) {
      List<String> names = rel.getTable().getQualifiedName();
      if (names != null) {
        //System.out.println("table name was: " + names.get(1));
        // TODO: is the more general version ever needed?
        //String tableName = "";
        //for (String s : names) {
          //tableName += s "-";
        //}
        //return tableName;
        return names.get(1);
      }
    }
    return null;
  }

  /* just a helper utility function that traverses the RelNode tree.
   */
  public static void printInfo(RelNode node) {
    Set<CorrelationId> setIds = node.getVariablesSet();
    System.out.println("num setIds: " + setIds.size());
    System.out.println("rel class: " + node.getClass().getName());
    System.out.println("rel convention: " + node.getConvention());
    System.out.println("rel query class: " + node.getQuery().getClass().getName());
    //System.out.println(RelOptUtil.toString(node));
    System.out.println("digest: " + node.recomputeDigest());
    RelDataType dt = node.getRowType();
    System.out.println("dt.toString: " + dt.toString());

    if (node instanceof LogicalJoin) {
        LogicalJoin lnode = (LogicalJoin) node;
        System.out.println("systemFieldList size: " + lnode.getSystemFieldList().size());
    }
    for (RelNode inp : node.getInputs()) {
        System.out.println("next input");
        printInfo(inp);
    }
  }

  /* FIXME: make this more flexible. Need to close ResultSet's so we just send
   * back the hash of top n rows if asked for.
   */
  public static class ExecutionResult {
    public Integer resultHashCode = -1;
    public Long runtime = 0L;
    public Double trueCardinality = -1.00;
  }

  private static void clearCache()
  {
      //System.out.println("clear cache...");
      try {
        String cmd = "./drop_cache.sh";
        Process cmdProc = Runtime.getRuntime().exec(cmd);
        cmdProc.waitFor();
        StringWriter writer = new StringWriter();
        InputStream inputStream = cmdProc.getInputStream();
        IOUtils.copy(inputStream, writer, StandardCharsets.UTF_8);
        String outString = writer.toString();
	System.out.println(outString);
        if (cmdProc.exitValue() != 0) {
          System.out.println(outString);
          System.out.println("Clearing cache failed. Exit value: " + cmdProc.exitValue());
          System.exit(-1);
        }
        // TODO: how long should we sleep for here to let postgres start fine?
        TimeUnit.MILLISECONDS.sleep(4000);
      } catch (Exception e) {
        System.out.println("trying to drop cache failed miserably");
        e.printStackTrace();
        System.exit(-1);
      }
  }

  /* Executes the given sql using plain old jdbc connection without calcite.
   * Postgres would do its own usual set of optimizations etc.
   * FIXME: try to decompose common parts with executeNode.
   */
  public static ExecutionResult executeSql(String sql,
                                           boolean getTrueCardinality,
                                           boolean clearCache)
  {
    QueryOptExperiment.Params params = QueryOptExperiment.getParams();
		ExecutionResult execResult = null;
    ResultSet rs = null;
		Connection con = null;
    PreparedStatement ps = null;

    if (clearCache) {
        clearCache();
    }
    try {
      Class.forName("org.postgresql.Driver");
      con = DriverManager.getConnection(params.pgUrl, params.user,
                                          params.pwd);
			//Statement stmt = con.createStatement();
      ps = con.prepareStatement(sql);
      ps.setQueryTimeout(params.maxExecutionTime);
      long start = System.currentTimeMillis();
      Long runtime = null;
      try {
        rs = ps.executeQuery();
      } catch (Exception e) {
        // do nothing, since this would be triggered by the queryTimeOut.
        System.out.println("queryTimeout!");
        runtime = (long) params.maxExecutionTime * 1000;
      }

      if (runtime == null) {
          long end = System.currentTimeMillis();
          runtime = end - start;
      }
        // this can be an expensive operation, so only do it if really needed.
      if ((params.verifyResults || getTrueCardinality) && rs != null) {
          execResult = getResultSetHash(rs);
          execResult.runtime = runtime;
      } else {
          // default values
          execResult = new ExecutionResult();
          execResult.runtime = runtime;
      }

    } catch (Exception e) {
      // TODO: this seems to fail sometimes if postgres hasn't started yet.
      // Handle that better instead of sleeping longer?
      e.printStackTrace();
      System.exit(-1);
    }

    try {
      con.close();
      ps.close();
      if (rs != null) rs.close();
    } catch (Exception e) {
      e.printStackTrace();
      // no good way to handle this (?)
      System.exit(-1);
    }

    return execResult;
  }

  /* @node: node to execute.
   * TODO: describe other params
   * TODO: remove dependence on QueryOpt.Params
   * @ret: ExecutionResult: ResultSet, ExecutionTime
   */
  public static ExecutionResult executeNode(RelNode node,
                                            boolean getTrueCardinality,
                                            boolean clearCache)
  {
    QueryOptExperiment.Params params = QueryOptExperiment.getParams();
    if (clearCache) {
        clearCache();
    }
    ResultSet rs = null;
    PreparedStatement ps = null;
    Integer resultHash = -1;
    Long runtime = null;
    CalciteConnection curConn = null;
    ExecutionResult execResult = null;

    try {
      curConn = (CalciteConnection) DriverManager.getConnection(params.dbUrl);
      curConn.setAutoCommit(true);
      RelRunner runner = curConn.unwrap(RelRunner.class);
      ps = runner.prepare(node);
      ps.setQueryTimeout(params.maxExecutionTime);
      long start = System.currentTimeMillis();
      runtime = null;

      try {
        rs = ps.executeQuery();
      } catch (Exception e) {
        // do nothing, since this would be triggered by the queryTimeOut.
        System.out.println("queryTimeout!");
        runtime = (long) params.maxExecutionTime * 1000;
      }

      if (runtime == null) {
          long end = System.currentTimeMillis();
          runtime = end - start;
      }
        // this can be an expensive operation, so only do it if really needed.
        if (params.verifyResults || getTrueCardinality) {
            execResult = getResultSetHash(rs);
            execResult.runtime = runtime;
        } else {
            // default values
            execResult = new ExecutionResult();
            execResult.runtime = runtime;
        }
    } catch (Exception e) {
      System.out.println("caught exception while executing query");
      StringWriter errors = new StringWriter();
      e.printStackTrace(new PrintWriter(errors));
      String errorMsg = errors.toString();
      // this error usually seems to happen in the execution attempt right
      // after clearing cache
      if (errorMsg.contains("administrator")) {
        System.out.println("contains: admin");
      }
      // this seems to happen when Avatica decides to mysteriously send a
      // cancelling by user request to psql (hypothesis: because it is taking
      // too long...)
      if (errorMsg.contains("user")) {
        System.out.println("contains: user");
      }

      if (!errorMsg.contains("user") && !errorMsg.contains("administrator")) {
        e.printStackTrace();
      }

      try {
        curConn.close();
        ps.close();
      } catch (Exception e2) {
        e2.printStackTrace();
        System.exit(-1);
      }
      return null;
    }
    /* clean up the remaining used resources */

    try {
      //TimeUnit.SECONDS.sleep(2);
      curConn.close();
      ps.close();
      if (rs != null) rs.close();
    } catch (Exception e) {
      e.printStackTrace();
      // no good way to handle this (?)
      System.exit(-1);
    }

    return execResult;
  }

  public static ExecutionResult getResultSetHash(ResultSet res)
  {
    String combinedResults = "";
    Double curLine = 0.00;
    try {
      ResultSetMetaData rmd = res.getMetaData();
      int num_columns = rmd.getColumnCount();
      System.out.println("num columns " + num_columns);
      while (res.next()) {
        // FIXME: do we want to go over all columns or not?
        for (int i = 1; i < num_columns+1; i++) {
          combinedResults += res.getString(i);
        }
        curLine += 1.00;
      }
    } catch (Exception e) {
      // ignore for now.
    }
    ExecutionResult execResult = new ExecutionResult();
    execResult.resultHashCode = combinedResults.hashCode();
    execResult.trueCardinality = curLine;
    return execResult;
  }
}
