package LDA;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

public class JavaLDAExample {

  public static void main(String[] args) {

    javaLDA();

  }

  public static void javaLDA() {

    SparkConf conf = new SparkConf().setAppName("LDA Example").setMaster(
        "local");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load and parse the data
    String path = "matrix/part-00000";
    JavaRDD<String> data = sc.textFile(path);

    JavaRDD<String> filtereddata = data.filter((String inLine) -> {
      if (inLine.length() > 2) {
        return true;
      }
      return false;
    });

    JavaRDD<Vector> parsedData = filtereddata
        .map(new Function<String, Vector>() {
          public Vector call(String s) {
            String[] sarray = s.trim().split(" ");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++)
              values[i] = Double.parseDouble(sarray[i]);
            return Vectors.dense(values);
          }
        });
    JavaLDAExample.saveRDDVectorAsHDFS(parsedData, "parsedData");

    // Index documents with unique IDs
    JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(parsedData
        .zipWithIndex().map(
            new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
              public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id) {
                return doc_id.swap();
              }
            }));
    corpus.cache();
    JavaLDAExample.saveRDDAsHDFS(corpus, "corpus");

    // Cluster the documents into three topics using LDA
    DistributedLDAModel ldaModel = new LDA().setK(3).run(corpus);

    // Output topics. Each is a distribution over words (matching word count
    // vectors)
    System.out.println("Learned topics (as distributions over vocab of "
        + ldaModel.vocabSize() + " words):");
    Matrix topics = ldaModel.topicsMatrix();
    for (int topic = 0; topic < 3; topic++) {
      System.out.print("Topic " + topic + ":");
      for (int word = 0; word < ldaModel.vocabSize(); word++) {
        System.out.print(" " + topics.apply(word, topic));
      }
      System.out.println();
    }

    String topicmatrix = "";
    for (int topic = 0; topic < 3; topic++) {
      System.out.print("Topic " + topic + ":");
      topicmatrix = topicmatrix + "Topic " + topic + ":";
      for (int word = 0; word < ldaModel.vocabSize(); word++) {
        System.out.print(" " + topics.apply(word, topic));
        topicmatrix = topicmatrix + " " + topics.apply(word, topic);
      }
      topicmatrix = topicmatrix + "\n";
      System.out.println();
    }

    saveRDDAsHDFS(sc.parallelize(Arrays.asList(topicmatrix)), "ldatopicmatrix");

    RDD<Tuple2<Object, Vector>> twittermatrix = ldaModel.topicDistributions();
    saveRDDAsHDFS(twittermatrix, "ldatwittermatrix");

  }

  public static void saveRDDAsHDFS(JavaRDD<String> tweets, String fileOut) {
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

  public static void saveRDDVectorAsHDFS(JavaRDD<Vector> tweets, String fileOut) {
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

  public static void saveRDDAsHDFS(JavaPairRDD<Long, Vector> tweets,
      String fileOut) {
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

  public static void saveRDDAsHDFS(RDD<Tuple2<Object, Vector>> tweets,
      String fileOut) {
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