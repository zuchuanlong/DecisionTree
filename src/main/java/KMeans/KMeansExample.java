package KMeans;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;

public class KMeansExample {

  public static void main(String[] args) {

    kMeans();

  }

  public static void kMeans() {

    SparkConf conf = new SparkConf().setAppName("K-means Example").setMaster(
        "local");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load and parse data
    String path = "/Users/chuanlongzu/Downloads/coords.txt";
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<Vector> parsedData = data.map(new Function<String, Vector>() {
      public Vector call(String s) {
        String[] sarray = s.split(" ");
        double[] values = new double[sarray.length];
        for (int i = 0; i < sarray.length; i++)
          values[i] = Double.parseDouble(sarray[i]);
        return Vectors.dense(values);
      }
    });
    parsedData.cache();

    // Cluster the data into two classes using KMeans
    int numClusters = 50;
    int numIterations = 20;
    KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters,
        numIterations);

    Vector[] v = clusters.clusterCenters();
    for (int n = 0; n < v.length; n++) {
      System.out.println(v[n]);
    }

    // saveRDDAsHDFS(clusters.predict(parsedData), "KMeans");

    saveRDDAsHDFS(
        clusters.predict(parsedData).map(new Function<Integer, String>() {
          public String call(Integer x) {
            return v[x].toString().replace("[", "").replace("]", "");
          }
        }), "KMeans");

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    double WSSSE = clusters.computeCost(parsedData.rdd());
    System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

  }

  protected static void saveRDDAsHDFS(JavaRDD<String> tweets, String fileOut) {
    try {
      URI fileOutURI = new URI(fileOut);
      URI hdfsURI = new URI(fileOutURI.getScheme(), null, fileOutURI.getHost(),
          fileOutURI.getPort(), null, null, null);
      Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
      FileSystem hdfs = org.apache.hadoop.fs.FileSystem
          .get(hdfsURI, hadoopConf);
      System.out.print(hdfsURI.toString());
      System.out.print(fileOutURI.toString());
      hdfs.delete(new org.apache.hadoop.fs.Path(fileOut), true);
      tweets.saveAsTextFile(fileOut);
    } catch (URISyntaxException | IOException e) {
      Logger.getRootLogger().error(e);
    }
  }

}