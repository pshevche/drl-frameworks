import java.util.*;
import java.io.*;
import org.apache.commons.io.FileUtils;
import com.google.gson.Gson;
import java.nio.file.*;
import java.nio.charset.*;
import com.google.gson.reflect.TypeToken;

public class Query {

  // public String fileName;
  public String queryName;
  public String sql;

  // FIXME: maybe instead of having a million maps, we just have a queryPlanner
  // object which keeps track of all this??
  public HashMap<String, Double> costs;
  //public HashMap<String, List<int[]>> joinOrders;
  public HashMap<String, Long> planningTimes;
  public HashMap<String, String> plans;
  public HashMap<String, Integer> resultVerifier;
  public HashMap<String, MyUtils.JoinOrder> joinOrders;

  // will represent runtimes by different algorithms on particular dbms'
  HashMap<String, Long> dbmsRuntimes;
  String allDBMSRuntimesFile = QueryOptExperiment.getParams().runtimeFileName;
  HashMap<String, ArrayList<Long>> dbmsAllRuntimes;
  // key will be the join order of the algorithm. Saving for each join order so
  // we can avoid re-running queries
  HashMap<List<Integer>, Double> RLRuntimes;

  // TODO: add alternative init methods
  public Query(String queryName, String querySql) throws Exception
  {
    this.queryName = queryName;
    this.sql = queryRewriteForCalcite(querySql);
    // initialize all the guys
    resultVerifier = new HashMap<String, Integer>();
    //plans = new HashMap<String, String>();
    //joinOrders = new HashMap<String, List<int[]>>();
    joinOrders = new HashMap<String, MyUtils.JoinOrder>();
    costs = new HashMap<String, Double>();
    plans = new HashMap<String, String>();
    planningTimes = new HashMap<String, Long>();
    HashMap<String, HashMap<String, ArrayList<Long>>> allQueryRuntimes
                        = loadDBMSRuntimes();
    if (allQueryRuntimes == null) {
      dbmsAllRuntimes = new HashMap<String, ArrayList<Long>>();
    } else {
      dbmsAllRuntimes = allQueryRuntimes.get(queryName);
    }
    if (dbmsAllRuntimes == null) {
      dbmsAllRuntimes = new HashMap<String, ArrayList<Long>>();
    }
    //queryName
    dbmsRuntimes = new HashMap<String, Long>();
    if (dbmsAllRuntimes.size() != 0) {
      // average of all keys
      for (String plannerName : dbmsAllRuntimes.keySet()) {
        Long total = 0L;
        ArrayList<Long> rts = dbmsAllRuntimes.get(plannerName);
        for (Long val : rts) {
          total += val;
        }
        dbmsRuntimes.put(plannerName, total / rts.size());
      }
    }
  }

  public HashMap<String, HashMap<String, ArrayList<Long>>> loadDBMSRuntimes()
  {
    Gson gson = new Gson();
    try {
      File f = new File(allDBMSRuntimesFile);
      if (f.exists()) {
        String jsonStr = new String(Files.readAllBytes(Paths.get(allDBMSRuntimesFile)),
                            StandardCharsets.UTF_8);
        return gson.fromJson(jsonStr,
            new TypeToken<HashMap<String, HashMap<String, ArrayList<Long>>>>() {}.getType());
      } else {
        //return new HashMap<String, ArrayList<Long>>();
        return null;
      }
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(-1);
    }
    return null;
  }

  public void saveDBMSRuntimes()
  {
    // load everything from disk, and then save it back
    HashMap<String, HashMap<String, ArrayList<Long>>> allQueryRuntimes
                        = loadDBMSRuntimes();
    if (allQueryRuntimes == null) {
      allQueryRuntimes = new HashMap<String, HashMap<String, ArrayList<Long>>>();
    }
    allQueryRuntimes.put(queryName, dbmsAllRuntimes);
    Gson gson = new Gson();
    String jsonStr = gson.toJson(allQueryRuntimes);
    try {
      PrintWriter writer = new PrintWriter(allDBMSRuntimesFile, "UTF-8");
      writer.println(jsonStr);
      writer.close();
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(-1);
    }
  }

  public String toJson() {
    Gson gson = new Gson();
    return gson.toJson(this);
  }

  private String queryRewriteForCalcite(String query) {
      String newQuery = query.replace(";", "");
      newQuery = newQuery.replace("!=", "<>");
      // weird trouble for calcite because of this
      newQuery = newQuery.replace("AS character", "");
      // using `at' for a table alias seems to cause trouble
      //newQuery = newQuery.replace("\\bat,\\b", "att,");
      newQuery = newQuery.replace("at,", "att,");
      newQuery = newQuery.replace("at.", "att.");
      // debugging purposes
      // FIXME: doesn't seem easy to add text here without running into
      // weird formatting issues (while it works just fine if we write the
      // same thing in the original queries)
      //newQuery = "\"explain\" " + newQuery;
      //newQuery = newQuery + " LIMIT 10";
      return newQuery;
  }

  public boolean verifyResults() {
    // need to check that all the values in res must be the same!
    Integer rl_val = resultVerifier.get("RL");
    for (Integer val : resultVerifier.values()) {
      if (!(val == rl_val)) {
        return false;
      }
    }
    return true;
  }
}
