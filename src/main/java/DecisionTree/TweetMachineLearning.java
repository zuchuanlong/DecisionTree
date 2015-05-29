package DecisionTree;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.json.JSONException;
import org.json.JSONObject;

import scala.Tuple2;
import au.org.aurin.tweetcommons.JavaRDDTweet;
import au.org.aurin.tweetcommons.Tweet;

public class TweetMachineLearning {

  public static void main(String args[]) {

    SparkConf conf = (new SparkConf()).setAppName("TweetML").setMaster("local");

    final JavaSparkContext sc = new JavaSparkContext(conf);

    JavaRDD<String> rawTweets = sc
        .textFile("hdfs://localhost:9000/data/byTimestamp.json");

    JavaRDD<String> filteredTweets = rawTweets.filter((String inLine) -> {
      try {
        JSONObject obj = new JSONObject(inLine);
        obj.getString("id");
      } catch (JSONException e) {
        return false;
      }
      return true;
    });

    JavaRDD<Tweet> parsedTweets = filteredTweets.map((String inLine) -> {
      return TweetCruncher.processTweet(new Tweet(new JSONObject(inLine)));
    });

    // Create the dictionary
    JavaRDD<String> dictionary = TweetCruncher.createDictionary(parsedTweets);
    JavaRDDTweet.saveRDDStringAsHDFS(dictionary,
        "hdfs://localhost:9000/data/dictionary.txt");

    /*
     * Create the hashmap of the whole tokens.
     * 
     * Key is one unique token.
     * 
     * Value is the line number of the token in the dictionary.
     */
    Iterator<String> dictionaryIte = dictionary.toLocalIterator();
    HashMap<String, Integer> dictionaryHM = new HashMap<String, Integer>();
    int dicLineNum = 0;

    while (dictionaryIte.hasNext()) {
      dictionaryHM.put(dictionaryIte.next(), dicLineNum);
      dicLineNum++;
    }

    // Broadcast the tokens hashmap.
    Broadcast<HashMap<String, Integer>> broadcastDic = sc
        .broadcast(dictionaryHM);

    // Create the tweets/words matrix.
    // Each row of the matrix is a sparse vector.
    JavaRDD<Vector> vectorMatrix = createMatrix(parsedTweets, broadcastDic);

    TweetMachineLearning.saveRDDVectorAsHDFS(vectorMatrix,
        "hdfs://localhost:9000/data/VectorMatrix.txt");

    // Create the LDA model
    int topicsNumber = 3;
    DistributedLDAModel ldaModel = createLDAModel(vectorMatrix, topicsNumber);

    // Create the topics matrix
    String topicMatrix = "";
    Matrix topics = ldaModel.topicsMatrix();
    JavaRDD<String> topicsMatrix = sc.parallelize(Arrays.asList(topicMatrix));
    for (int topic = 0; topic < topicsNumber; topic++) {
      topicMatrix = "Topic " + topic + ":";
      for (int word = 0; word < ldaModel.vocabSize(); word++) {
        topicMatrix = topicMatrix + " " + topics.apply(word, topic);
      }
      topicsMatrix = topicsMatrix.union(sc.parallelize(Arrays
          .asList(topicMatrix)));
      topicMatrix = "";
    }

    saveRDDStringAsHDFS(topicsMatrix,
        "hdfs://localhost:9000/data/TopicsMatrix.txt");

    // Create the topic distributions
    RDD<Tuple2<Object, Vector>> topicDistributions = ldaModel
        .topicDistributions();
    saveRDDTupleAsHDFS(topicDistributions,
        "hdfs://localhost:9000/data/TopicDistributions.txt");

  }

  /*
   * Create the tweets/words matrix used for the input in LDA.
   * 
   * Each row of the matrix is a sparse vector to save storage.
   * 
   * @param parsedTweets: The Tweets have been parsed containing tokens.
   * 
   * @param dictionary: The hashmap contains the whole tokens.
   * 
   * @return the tweets/words matrix.
   */
  public static JavaRDD<Vector> createMatrix(JavaRDD<Tweet> parsedTweets,
      Broadcast<HashMap<String, Integer>> dictionary) {

    HashMap<String, Integer> dictionaryHM = dictionary.value();

    JavaRDD<Vector> vectorMatrix = parsedTweets.map((Tweet tweet) -> {

      List<String> tokenList = tweet.getTokens();
      int[] tokenIndex = new int[tokenList.size()];
      double[] tokenValue = new double[tokenList.size()];

      for (int n = 0; n < tokenList.size(); n++) {
        tokenIndex[n] = dictionaryHM.get(tokenList.get(n));
        tokenValue[n] = Double.parseDouble("1");
      }

      return Vectors.sparse(dictionaryHM.size(), tokenIndex, tokenValue);

    });

    return vectorMatrix;

  }

  /*
   * Create LDA model to calculate the topics matix and the topic distributions
   * 
   * @param vectorMatrix: The tweets/words matrix.
   * 
   * @param topicsNum: The number of topics.
   * 
   * @return the LDA model.
   */
  public static DistributedLDAModel createLDAModel(
      JavaRDD<Vector> vectorMatrix, int topicsNum) {

    JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(vectorMatrix
        .zipWithIndex().map((Tuple2<Vector, Long> doc_id) -> {
          return doc_id.swap();
        }));
    corpus.cache();

    DistributedLDAModel ldaModel = new LDA().setK(topicsNum).run(corpus);

    return ldaModel;

  }

  /*
   * The functions below are not moved to tweetcommons yet for convenience.
   * 
   * When complete the machine learning, these functions will be moved to
   * tweetcommons together.
   */

  public static void saveRDDVectorAsHDFS(JavaRDD<Vector> rddIn, String fileOut) {
    try {
      URI fileOutURI = new URI(fileOut);
      URI hdfsURI = new URI(fileOutURI.getScheme(), null, fileOutURI.getHost(),
          fileOutURI.getPort(), null, null, null);
      Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
      FileSystem hdfs = org.apache.hadoop.fs.FileSystem
          .get(hdfsURI, hadoopConf);
      hdfs.delete(new org.apache.hadoop.fs.Path(fileOut), true);
      rddIn.saveAsTextFile(fileOut);
    } catch (URISyntaxException | IOException e) {
      Logger.getRootLogger().error(e);
    }
  }

  public static void saveRDDStringAsHDFS(JavaRDD<String> rddIn, String fileOut) {
    try {
      URI fileOutURI = new URI(fileOut);
      URI hdfsURI = new URI(fileOutURI.getScheme(), null, fileOutURI.getHost(),
          fileOutURI.getPort(), null, null, null);
      Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
      FileSystem hdfs = org.apache.hadoop.fs.FileSystem
          .get(hdfsURI, hadoopConf);
      hdfs.delete(new org.apache.hadoop.fs.Path(fileOut), true);
      rddIn.saveAsTextFile(fileOut);
    } catch (URISyntaxException | IOException e) {
      Logger.getRootLogger().error(e);
    }
  }

  public static void saveRDDTupleAsHDFS(RDD<Tuple2<Object, Vector>> rddIn,
      String fileOut) {
    try {
      URI fileOutURI = new URI(fileOut);
      URI hdfsURI = new URI(fileOutURI.getScheme(), null, fileOutURI.getHost(),
          fileOutURI.getPort(), null, null, null);
      Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
      FileSystem hdfs = org.apache.hadoop.fs.FileSystem
          .get(hdfsURI, hadoopConf);
      hdfs.delete(new org.apache.hadoop.fs.Path(fileOut), true);
      rddIn.saveAsTextFile(fileOut);
    } catch (URISyntaxException | IOException e) {
      Logger.getRootLogger().error(e);
    }
  }

}
