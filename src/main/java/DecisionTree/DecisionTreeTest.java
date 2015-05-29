package DecisionTree;

import java.io.IOException;
import java.math.BigDecimal;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.json.JSONObject;

import scala.Tuple2;

public class DecisionTreeTest {

  public static void main(String args[]) {

    // Initial 
    SparkConf sparkConf = new SparkConf().setAppName("DataFrame").setMaster(
        "local");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    JavaRDD<String> weather = sc
        .textFile("hdfs://localhost:9000/data/weather.json");

    List<String> keyList = weather.map(
        (String s) -> {
          JSONObject key = new JSONObject(s);
          return createKey(key.get("timestamp").toString(), key.get("location")
              .toString());
        }).collect();

    // Discretizing the continuous weather features and getting the cluster
    // indices of each feature.
    List<String> temList = createKMeansWeather(weather, "mintmp", 5);
    List<String> humidList = createKMeansWeather(weather, "hum", 5);
    List<String> windList = createKMeansWeather(weather, "wind", 5);
    List<String> cloudList = createKMeansWeather(weather, "rain", 5);
    List<String> rainList = createKMeansWeather(weather, "press", 5);

    /*
     * Create the hashmap for the weather.
     * 
     * Key is the combination of location and timestamp.
     * 
     * Value is the double class array containing the clusters indices.
     */
    HashMap<String, double[]> weatherHM = new HashMap<>();
    // double[] weatherArray = new double[5];

    for (int n = 0; n < temList.size(); n++) {
      double[] weatherArray = new double[5];
      for (int m = 0; m < 5; m++) {
        switch (m) {
        case 0:
          weatherArray[m] = cutDouble(temList.get(n));
          break;
        case 1:
          weatherArray[m] = cutDouble(humidList.get(n));
          break;
        case 2:
          weatherArray[m] = cutDouble(windList.get(n));
          break;
        case 3:
          weatherArray[m] = cutDouble(cloudList.get(n));
          break;
        case 4:
          weatherArray[m] = cutDouble(rainList.get(n));
          break;
        }
      }
      // System.out.println(weatherArray[0]);
      weatherHM.put(keyList.get(n), weatherArray);

    }

    JavaRDD<String> parsedTweets = sc
        .textFile("hdfs://localhost:9000/data/Twitters.json");

    // Broadcast the weather hashmap
    Broadcast<HashMap<String, double[]>> broadcastWeather = sc
        .broadcast(weatherHM);

    // Create the decision tree model
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
    categoricalFeaturesInfo.put(0, 5);
    categoricalFeaturesInfo.put(1, 5);
    categoricalFeaturesInfo.put(2, 5);
    categoricalFeaturesInfo.put(3, 5);
    categoricalFeaturesInfo.put(4, 5);
    // Create the training data from Tweets and weather for building the
    // decision tree model
    // createTW(parsedTweets, broadcastWeather).saveAsTextFile("TW");
    decisionTreeRegression(createTW(parsedTweets, broadcastWeather),
        categoricalFeaturesInfo, 30, 32);

  }

  /*
   * Retain one digit after the decimal point.
   * 
   * @param s: The number to be processed.
   * 
   * @return the double retaining one digit after the decimal point.
   */
  public static Double cutDouble(String s) {
    return new BigDecimal(Double.valueOf(s)).setScale(1,
        BigDecimal.ROUND_HALF_UP).doubleValue();
  }

  /*
   * Discretizing the weather continuous features using the KMeans algorithm.
   * 
   * @param weather: The raw weather data.
   * 
   * @param key: The specific continuous feature.
   * 
   * @param cluster: The number of desired clusters.
   * 
   * @return the list of cluster indices for the weather feature.
   */
  public static List<String> createKMeansWeather(JavaRDD<String> weather,
      String key, int cluster) {

    // Pick up the specific attribute from weather
    JavaRDD<String> attribute = weather.map((s) -> {
      try {
        JSONObject obj = new JSONObject(s);
        return obj.get(key).toString();
      } catch (Exception e) {
        e.printStackTrace();
        return null;
      }
    });

    // KMeans
    JavaRDD<Vector> parsedData = attribute.map((s) -> {
      return Vectors.dense(Double.parseDouble(s));
    });
    parsedData.cache();

    int numIterations = 20;
    KMeansModel clusters = KMeans.train(parsedData.rdd(), cluster,
        numIterations);

    /*
     * Vector[] v = clusters.clusterCenters(); return
     * clusters.predict(parsedData).map((x) -> { return
     * v[x].toString().replace("[", "").replace("]", ""); }).collect();
     */

    return clusters.predict(parsedData).map((x) -> {
      return x.toString();
    }).collect();

  }

  /*
   * Create the unique key based on the timestamp and the location.
   * 
   * @param TimeStamp: The timestamp information in JSON.
   * 
   * @param Location: The location information in JSON.
   * 
   * @return the unique key used in the hashmap.
   */
  public static String createKey(String TimeStamp, String Location) {

    return TimeStamp + Location;

  }

  /*
   * Create the training data of Tweets and weather used for the input in
   * creating the decision tree.
   * 
   * @param parsedTweets: The Tweets which have been parsed.
   * 
   * @param broadcastWeather: The hashmap containing weather data.
   * 
   * @return the training data.
   */
  public static JavaRDD<LabeledPoint> createTW(JavaRDD<String> parsedTweets,
      Broadcast<HashMap<String, double[]>> broadcastWeather) {

    HashMap<String, double[]> weather = broadcastWeather.value();

    JavaRDD<LabeledPoint> data = parsedTweets.map((String tweet) -> {
      JSONObject tweets = new JSONObject(tweet);
      /*
       * System.out.println(createKey(tweets.get("timestamp").toString(), tweets
       * .get("location").toString()));
       * System.out.println(weather.get(createKey(tweets.get("timestamp")
       * .toString(), tweets.get("location").toString()))[0]);
       */
      return new LabeledPoint(Double.parseDouble(tweets.get("sentiment")
          .toString()), Vectors.dense(weather
          .get(createKey(tweets.get("timestamp").toString(),
              tweets.get("location").toString()))));
    });
    return data;

  }

  /*
   * Create the decision tree model(regression)
   * 
   * @param data: The JavaRDD<LabeledPoint> created from tweets and weather
   * 
   * @param categoricalFeaturesInfo: Specifies which features are categorical
   * and how many categorical values each of those features can take.
   * 
   * @param maxDepth: Maximum depth of a tree.
   * 
   * @param maxBins: Number of bins used when discretizing continuous features.
   * 
   * @return the decision tree model.
   */
  public static DecisionTreeModel decisionTreeRegression(
      JavaRDD<LabeledPoint> data,
      Map<Integer, Integer> categoricalFeaturesInfo, int maxDepth, int maxBins) {

    JavaRDD<LabeledPoint>[] splits = data
        .randomSplit(new double[] { 0.7, 0.3 });
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    String impurity = "variance";

    final DecisionTreeModel model = DecisionTree.trainRegressor(trainingData,
        categoricalFeaturesInfo, impurity, maxDepth, maxBins);

    // Evaluate model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel = testData
        .mapToPair((LabeledPoint p) -> {
          return new Tuple2<Double, Double>(model.predict(p.features()), p
              .label());
        });
    Double testMSE = predictionAndLabel.map((Tuple2<Double, Double> pl) -> {
      Double diff = pl._1() - pl._2();
      return diff * diff;
    }).reduce((Double a, Double b) -> {
      return a + b;
    }) / data.count();
    System.out.println("Test Mean Squared Error: " + testMSE);
    System.out.println("Learned regression tree model:\n"
        + model.toDebugString());

    return model;

  }

  /*
   * Save the decision tree model.
   * 
   * @param model: The decision tree model.
   * 
   * @param sc: The SparkContext.
   * 
   * @param fileout: The path where saving the decision tree model.
   */
  public static void saveDTModelAsHDFS(DecisionTreeModel model,
      SparkContext sc, String fileOut) {
    try {
      URI fileOutURI = new URI(fileOut);
      URI hdfsURI = new URI(fileOutURI.getScheme(), null, fileOutURI.getHost(),
          fileOutURI.getPort(), null, null, null);
      Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
      FileSystem hdfs = org.apache.hadoop.fs.FileSystem
          .get(hdfsURI, hadoopConf);
      hdfs.delete(new org.apache.hadoop.fs.Path(fileOut), true);
      model.save(sc, fileOut);
    } catch (URISyntaxException | IOException e) {
      Logger.getRootLogger().error(e);
    }
  }

}