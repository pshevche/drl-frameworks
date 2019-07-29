import java.sql.*;
import java.util.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.tools.*;
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
import java.io.File;
import java.util.concurrent.ThreadLocalRandom;
// experimental:
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.util.ImmutableBitSet;
import java.util.concurrent.TimeUnit;
import org.apache.commons.io.FileUtils;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

//import org.apache.calcite.rel.rel2sql.RelToSqlConverter;
//import org.apache.calcite.rel.RelToSqlConverter;
//import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.dialect.AnsiSqlDialect;
import org.apache.calcite.rel.rel2sql.SqlImplementor.Result;
import org.apache.calcite.sql.SqlNode;

/* Will contain all the parameters / data etc. to drive one end to end
 * experiment.
 */
public class QueryOptExperiment {

  private static CalciteConnection conn;
  private static String dbUrl;

  public enum PLANNER_TYPE
  {
    EXHAUSTIVE,
    LOpt,
    RANDOM,
    BUSHY,
    RL,
    LEFT_DEEP,
    GREEDY;

    // FIXME: not sure if we need to add other rules - like we
    // could add all of the Programs.RULE_SET here, and remove the
    // exhaustive rules above (that was done in heuristicJoinOrder)
    public static final ImmutableList<RelOptRule> EXHAUSTIVE_RULES =
        ImmutableList.of(ExhaustiveDPJoinOrderRule.INSTANCE,
                         FilterJoinRule.FILTER_ON_JOIN,
                         ProjectMergeRule.INSTANCE);
    public static final ImmutableList<RelOptRule> LEFT_DEEP_RULES =
        ImmutableList.of(LeftDeepJoinOrderRule.INSTANCE,
                         FilterJoinRule.FILTER_ON_JOIN,
                         ProjectMergeRule.INSTANCE);
    public static final ImmutableList<RelOptRule> LOPT_RULES =
        ImmutableList.of(MyLoptOptimizeJoinRule.INSTANCE,
                         FilterJoinRule.FILTER_ON_JOIN,
                         ProjectMergeRule.INSTANCE);
    // Note: need the second projection rule as otherwise the optimized
    // node from the joins was projecting all the fields before projecting
    // it down to only the selected fields
    public static final ImmutableList<RelOptRule> RL_RULES =
        ImmutableList.of(RLJoinOrderRule.INSTANCE,
                         FilterJoinRule.FILTER_ON_JOIN,
                         ProjectMergeRule.INSTANCE);

    public static final ImmutableList<RelOptRule> BUSHY_RULES =
        ImmutableList.of(MultiJoinOptimizeBushyRule.INSTANCE,
                        FilterJoinRule.FILTER_ON_JOIN,
                        ProjectMergeRule.INSTANCE);
    /// FIXME: implement these!!
    public static final ImmutableList<RelOptRule> RANDOM_RULES =
        ImmutableList.of(MultiJoinOptimizeBushyRule.INSTANCE);
    public static final ImmutableList<RelOptRule> GREEDY_RULES =
        ImmutableList.of(MultiJoinOptimizeBushyRule.INSTANCE);

    public ImmutableList<RelOptRule> getRules() {
      switch(this){
        case EXHAUSTIVE:
          return EXHAUSTIVE_RULES;
        case LOpt:
          return LOPT_RULES;
        case RANDOM:
          return RANDOM_RULES;
        case BUSHY:
          return BUSHY_RULES;
        case RL:
          return RL_RULES;
        case LEFT_DEEP:
          return LEFT_DEEP_RULES;
        default:
          return null;
      }
    }
  }
  private ArrayList<PLANNER_TYPE> plannerTypes;
  /* actual volcanoPlanners generated using the above rules */
  private ArrayList<Planner> volcanoPlanners;

  /* keeps all the pluggable parameters for the experiment, mostly set through
   * the command line parsing in Main.java */
  public static class Params {

    // initalize these with their default values incase someone doesn't supply
    // them

    // If execOnDB is false, and onlyFinalReward is true, then we will
    // treat the final reward as a sum of all the intermediate rewards.
    //public boolean onlyFinalReward = false;
    public boolean execOnDB = false;
    public boolean verifyResults = false;
    public boolean recomputeFixedPlanners = true;
    public Integer maxExecutionTime = 1200;
    public boolean python = true;
    public String dbUrl = "";
    // FIXME: make this command line arg
    public String pgUrl = "jdbc:postgresql://localhost:5400/imdb";
    public String user = "imdb";
    public String pwd = "imdb";
    // clear cache after every execution
    public boolean clearCache = false;
    public String cardinalitiesModel = "file";
    public String cardinalitiesModelFile = "./pg.json";
    //public String cardinalityError = "noError";
    //public Integer cardErrorRange = 10;
    // num reps for runtimes
    public Integer numExecutionReps = 1;
    public boolean train = false;
    public String runtimeFileName = "allQueryRuntimes.json";
    // cardinalities dictionary.
    public HashMap<String, HashMap<String, Long>> cardinalities = null;

    public Params() {

    }
  }

  private static Params params;

  // FIXME: move all these to Params
  public static ZeroMQServer zmq;

  // command line flags, parsed by Main.java. See the definitions / usage
  // there.
  private static String costModelName;
  //private static boolean onlyFinalReward;
  private static boolean verbose;
  private static boolean train;
  private static Query currentQuery;

  public static ArrayList<Query> trainQueries = null;
  public static ArrayList<Query> testQueries = null;
  // testing if features were set correctly
  public MyMetadataQuery mq;

  /*
  *************************************************************************
  *************************************************************************
                                  Methods
  *************************************************************************
  *************************************************************************
  */

  /* @dbUrl
   * @plannerTypes
   * @dataset
   */
  public QueryOptExperiment(String dbUrl, ArrayList<PLANNER_TYPE>
      plannerTypes, int port,
      boolean verbose, String costModelName,
      Params params) throws SQLException
  {
    this.params = params;
    // starts in training mode
    //this.train = true;
    this.costModelName = costModelName;
    this.verbose = verbose;
    this.dbUrl = dbUrl;
    this.conn = (CalciteConnection) DriverManager.getConnection(dbUrl);
    DbInfo.init(conn);
    this.zmq = new ZeroMQServer(port, verbose);
    this.plannerTypes = plannerTypes;
    volcanoPlanners = new ArrayList<Planner>();
    this.mq = MyMetadataQuery.instance();

    // Initialize all the volcanoPlanners we should need
    for (PLANNER_TYPE t  : plannerTypes) {
      Frameworks.ConfigBuilder bld = getDefaultFrameworkBuilder();
      bld.programs(MyJoinUtils.genJoinRule(t.getRules(), 1));
      Planner planner = Frameworks.getPlanner(bld.build());
      volcanoPlanners.add(planner);
    }

    // let's initialize cardinalities dictionary to given file
    if (params.cardinalitiesModel.equals("file")) {
      File file = new File(params.cardinalitiesModelFile);
      String jsonStr = null;
      try {
        jsonStr = FileUtils.readFileToString(file, "UTF-8");
      } catch (Exception e) {
        e.printStackTrace();
        System.exit(-1);
      }
      setCardinalities(jsonStr);
    }
  }

  // static setter functions
  public static void setTrainMode(boolean mode) {
    params.train = mode;
  }

  // static getter functions.
  // FIXME: explain why we have so many of these
  public static String getCostModelName() {
    return costModelName;
  }

  public static CalciteConnection getCalciteConnection() {
    return conn;
  }

  public static Params getParams() {
    return params;
  }

  public static Query getCurrentQuery() {
    return currentQuery;
  }

  public static ZeroMQServer getZMQServer() {
    return zmq;
  }

  public static void setQueries(String mode, HashMap<String, String> queries)
  {
    ArrayList<Query> queryList = null;
    if (mode.equals("train")) {
      trainQueries = new ArrayList<Query>();
      queryList = trainQueries;
    } else {
      testQueries = new ArrayList<Query>();
      queryList = testQueries;
    }
    try {
      for (String queryName : queries.keySet()) {
        queryList.add(new Query(queryName, queries.get(queryName)));
      }
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(-1);
    }
    if (verbose) System.out.println("successfully setQueries!");
  }

  /* FIXME: finalize queries semantics AND write explanation.
   * FIXME: used for both train and test, change name to reflect that.
   */
  public void start() throws Exception
  {
    // we will treat queries as a pool of sample data. After every reset, we
    // choose a new
    int numSuccessfulQueries = 0;
    int numFailedQueries = 0;
    // start a server, and wait for a command.
    if (params.python) zmq.waitForClientTill("getAttrCount");
    int nextQuery = -1;
    ArrayList<Query> queries;
    // TODO: ugh clean up the way i handle ordering etc.
    boolean alreadyTesting = false;
    while (true) {
      if (trainQueries == null) {
        zmq.waitForClientTill("setTrainQueries");
      }

      if (testQueries == null) {
        zmq.waitForClientTill("setTestQueries");
      }

      // at this point, all the other planners would have executed on the
      // current query as well, so all the stats about it would be updated in
      // the Query struct.
      if (params.python) zmq.waitForClientTill("getQueryInfo");
      if (params.python) zmq.waitForClientTill("reset");

      if (params.train) {
        alreadyTesting = false;
        queries = trainQueries;
        // FIXME: is deterministic order ok always?
        nextQuery = (nextQuery + 1) % queries.size();
      } else {
        queries = testQueries;
        if (alreadyTesting) {
          nextQuery = (nextQuery + 1) % queries.size();
        } else {
          nextQuery = 0;
          alreadyTesting = true;
        }
      }
      // FIXME: simplify this
      Query query = queries.get(nextQuery);

      if (verbose) System.out.println("nextQuery is: " + nextQuery);

      String sqlQuery = query.sql;
      currentQuery = query;
      zmq.sqlQuery = sqlQuery;
      if (zmq.END) break;
      if (volcanoPlanners.size() == 0) {
        System.out.println("no planners specified");
        break;
      }
      zmq.reset = false;
      //System.out.println("next query: " + query.queryName);
      for (int i = 0; i < volcanoPlanners.size(); i++) {
        try {
          boolean success = planAndExecuteQuery(query, i);
        } catch (Exception e) {
          String plannerName = plannerTypes.get(i).name();
          e.printStackTrace();
          query.costs.put(plannerName, -1.00);
          System.out.println("failed in planAndExecute for " + plannerName +
              " for query number " + nextQuery);
          System.exit(-1);
        }
      }
      if (params.verifyResults) {
        if (!query.verifyResults()) {
          System.err.println("verifying results failed");
          // just exit because what else to do?
          System.exit(-1);
        }
      }
    }
  }

  private RelOptCost computeCost(RelMetadataQuery mq, RelNode node) {
    return ((MyMetadataQuery) mq).getCumulativeCost(node);
  }

  private void execPlannerOnDB(Query query, String plannerName, RelNode node)
  {
    // run unoptimized sql query. This should use postgres' usual
    // optimizations.
    ArrayList<Long> savedRTs = query.dbmsAllRuntimes.get(plannerName);
    if (savedRTs == null) {
      savedRTs = new ArrayList<Long>();
    }

    if (savedRTs.size() >= params.numExecutionReps
          && !plannerName.equals("RL")) {
      // don't re-execute and waste everyone's breathe
      return;
    }
    MyUtils.ExecutionResult result = null;
    // run it one extra time
    for (int i = 0; i < 3; i++) {
      if (plannerName.equals("postgres")) {
        // run N times, and store the average
        result = MyUtils.executeSql(query.sql, false,
            params.clearCache);
      } else {
        result = MyUtils.executeNode(node, false, params.clearCache);
      }
      System.out.println(plannerName + " took " + result.runtime + "ms" + " for " + query.queryName);
      savedRTs.add(result.runtime);
    }
    query.dbmsAllRuntimes.put(plannerName, savedRTs);
    query.dbmsRuntimes.put(plannerName, result.runtime);
    query.saveDBMSRuntimes();
  }

  private boolean planAndExecuteQuery(Query query, int plannerNum)
    throws Exception
  {
    Planner planner = volcanoPlanners.get(plannerNum);
    String plannerName = plannerTypes.get(plannerNum).name();
    //System.out.println("planAndExecuteQuery with: " + plannerName);
    // Need to do this every time we reuse a planner
    planner.close();
    planner.reset();
    // first, have we already run this planner + query combination before?
    // In that case, we have no need to execute it again, as the result
    // will be stored in the Query object. Never do caching for RL since
    // the plans are constantly changing.
    if (!plannerName.equals("RL") &&
            !params.recomputeFixedPlanners
            && !params.execOnDB)
    {
      Double cost = query.costs.get(plannerName);
      // if we already executed this query, and nothing should change for the
      // deterministic planners
      if (cost != null) return true;
    }
    RelNode node = null;
    try {
      SqlNode sqlNode = planner.parse(query.sql);
      SqlNode validatedSqlNode = planner.validate(sqlNode);
      node = planner.rel(validatedSqlNode).project();
    } catch (Exception e) {
      System.out.println(e);
      System.out.println("failed in getting Relnode from  " + query.sql);
      return false;
      //System.exit(-1);
    }

    DbInfo.setCurrentQueryVisibleFeatures(node);
    // very important to do the replace EnumerableConvention thing for
    // mysterious reasons
    RelTraitSet traitSet = planner.getEmptyTraitSet().replace(EnumerableConvention.INSTANCE);
    // experimental:
    //RelTraitSet traitSet = planner.getEmptyTraitSet().replace(Convention.NONE);
    //RelTraitSet traitSet = null;
    //RelTraitSet traitSet = planner.getEmptyTraitSet();

    try {
      // using the default volcano planner.
      long start = System.currentTimeMillis();
      String origPlan = RelOptUtil.dumpPlan("", node, SqlExplainFormat.TEXT, SqlExplainLevel.ALL_ATTRIBUTES);
      //System.out.println(origPlan);
      RelNode optimizedNode = planner.transform(0, traitSet,
              node);
      long planningTime = System.currentTimeMillis() - start;
      if (verbose) System.out.println("planning time: " +
          planningTime);
      RelOptCost optCost = computeCost(mq, optimizedNode);
      // Time to update the query with the current results
      query.costs.put(plannerName, ((MyCost) optCost).getCost());
      if (verbose) System.out.println("optimized cost for " + plannerName
          + " is: " + optCost);
      //System.out.println("optimized cost for " + plannerName + " is: " + optCost);

      String optPlan = RelOptUtil.dumpPlan("", optimizedNode, SqlExplainFormat.TEXT, SqlExplainLevel.ALL_ATTRIBUTES);
      query.plans.put(plannerName, optPlan);
      // slightly complicated because we need information from multiple places
      // in the joinOrder struct
      MyUtils.JoinOrder joinOrder = query.joinOrders.get(plannerName);
      joinOrder = MyUtils.updateJoinOrder(optimizedNode, joinOrder);
      query.joinOrders.put(plannerName, joinOrder);
      query.planningTimes.put(plannerName, planningTime);
      if (params.execOnDB) {
        if (!params.train) {
          execPlannerOnDB(query, plannerName, optimizedNode);
        } else if (plannerName.equals("RL")) {
          execPlannerOnDB(query, plannerName, optimizedNode);
        }
      }
    } catch (Exception e) {
      // it is useful to throw the error here to see what went wrong..
      throw e;
    }

    // FIXME: to execute on postgres, just execute plain sql
    if (params.execOnDB && !params.train) {
      execPlannerOnDB(query, "postgres", node);
    }
    return true;
  }

  private Frameworks.ConfigBuilder getDefaultFrameworkBuilder() throws
      SQLException
  {
    // build a FrameworkConfig using defaults where values aren't required
    Frameworks.ConfigBuilder configBuilder = Frameworks.newConfigBuilder();
    configBuilder.defaultSchema(conn.getRootSchema().getSubSchema(conn.getSchema()));
    // seems the simplest to get it working. Look at older commits for
    // other variants on this
    configBuilder.parserConfig(SqlParser.configBuilder()
                                .setLex(Lex.MYSQL)
                                .build());
    // FIXME: not sure if all of these are required?!
    final List<RelTraitDef> traitDefs = new ArrayList<RelTraitDef>();
    traitDefs.add(ConventionTraitDef.INSTANCE);
    traitDefs.add(RelCollationTraitDef.INSTANCE);
    configBuilder.traitDefs(traitDefs);
    configBuilder.context(Contexts.EMPTY_CONTEXT);
    configBuilder.costFactory(MyCost.FACTORY);
    return configBuilder;
  }

  /* @node: the node to be executed over the jdbc connection (this.conn).
   * @ret: The cardinality of the result.
   * Note: this has ONLY been tested when node is a base table relation (SCAN or
   * FILTER->SCAN on a table)
   * FIXME: This seems to fail if the query takes too long to execute...
   * (e.g., SELECT * from cast_info, or SELECT * from movie_info)
   */
  public static Double getTrueCardinality(RelNode node)
  {
    CalciteConnection curConn;
    PreparedStatement ps = null;
    ResultSet res = null;
    Double cardinality = null;
    try {
      // recreating the connection should also work equally well.
      //curConn = (CalciteConnection) DriverManager.getConnection(dbUrl);
      curConn = conn;
      RelRunner runner = curConn.unwrap(RelRunner.class);
      ps = runner.prepare(node);
      long start = System.currentTimeMillis();
      res = ps.executeQuery();
      long end = System.currentTimeMillis();
      long total = end - start;
      System.out.println("execution time: " + total);
      if (res != null) {
        cardinality = 0.00;
        while (res.next()) {
          cardinality += 1.00;
        }
      } else {
        // something went wrong? should we fail?
        System.err.println("something went wrong while computing cardinality!!!");
        cardinality = 0.00;
      }
    } catch (SQLException e) {
      System.out.println("caught exception while trying to find cardinality of subquery");
      System.out.println(e);
      e.printStackTrace();
      //System.exit(-1);
      // FIXME: temp.
      return 100.00;
    }
    try {
      ps.close();
      res.close();
    } catch (Exception e) {
      System.out.println(e);
      e.printStackTrace();
      // no good way to handle this (?)
      //System.exit(-1);
      // FIXME: temp.
      return 100.00;
    }
    if (verbose) System.out.println("true cardinality was: " + cardinality);
    return cardinality;
  }

  public static void setCardinalities(String jsonStr) {
      Gson gson = new Gson();
      params.cardinalities = gson.fromJson(jsonStr,
          new TypeToken<HashMap<String, HashMap<String, Long>>>() {}.getType());

  }
}
