package DecisionTree;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.json.JSONObject;

import au.org.aurin.tweetcommons.Tweet;

public class DecisionTree {

  public static void main(String args[]) {

    SparkConf sparkConf = new SparkConf().setAppName("DataFrame").setMaster(
        "local");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    JavaRDD<String> weather = sc
        .textFile("hdfs://localhost:9000/data/weather.json");

    List<String> temList = createKMeansWeather(weather, "temperature", 5);
    List<String> humidList = createKMeansWeather(weather, "humidity", 5);
    List<String> windList = createKMeansWeather(weather, "wind", 5);
    List<String> cloudList = createKMeansWeather(weather, "cloud", 5);
    List<String> rainList = createKMeansWeather(weather, "rain", 5);

    HashMap<String, String> weatherHM = new HashMap<>();
    String[] weatherArray = new String[temList.size()];

    for (int n = 0; n < temList.size(); n++) {
      for (int m = 0; m < 5; m++) {
        switch (m) {
        case 0:
          weatherArray[n] = "1:" + cutDouble(temList.get(n));
          break;
        case 1:
          weatherArray[n] = weatherArray[n] + " 2:"
              + cutDouble(humidList.get(n));
          break;
        case 2:
          weatherArray[n] = weatherArray[n] + " 3:"
              + cutDouble(windList.get(n));
          break;
        case 3:
          weatherArray[n] = weatherArray[n] + " 4:"
              + cutDouble(cloudList.get(n));
          break;
        case 4:
          weatherArray[n] = weatherArray[n] + " 5:"
              + cutDouble(rainList.get(n));
          break;
        }
      }
      weatherHM.put(String.valueOf(n), weatherArray[n]);
    }

    for (int n = 0; n < temList.size(); n++) {

      System.out.println(weatherArray[n]);

    }

    Broadcast<HashMap<String, String>> broadcastWeather = sc
        .broadcast(weatherHM);

  }

  public static String cutDouble(String s) {
    return new BigDecimal(Double.valueOf(s)).setScale(1,
        BigDecimal.ROUND_HALF_UP).toString();
  }

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

    Vector[] v = clusters.clusterCenters();

    return clusters.predict(parsedData).map((x) -> {
      return v[x].toString().replace("[", "").replace("]", "");
    }).collect();

  }

  public static void createSVM(JavaRDD<Tweet> parsedTweets,
      Broadcast<HashMap<String, String>> broadcastWeather) {

  }

}
