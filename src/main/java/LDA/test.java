package LDA;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import scala.Tuple2;
import tachyon.thrift.WorkerService.Processor.returnSpace;

public class test {

  public static void main(String args[]) {

    SparkConf conf = new SparkConf().setAppName("LDA Example").setMaster(
        "local");
    JavaSparkContext sc = new JavaSparkContext(conf);

    JavaRDD<String> twitter = sc
        .textFile("hdfs://localhost:9000/data/Twitters.json");
    JavaRDD<String> dictionary = sc
        .textFile("hdfs://localhost:9000/data/dictionary.txt");

    JavaPairRDD<String, Integer> pairs = dictionary
        .mapToPair(new PairFunction<String, String, Integer>() {
          int n = 0;

          public Tuple2<String, Integer> call(String s) {
            n = n++;
            return new Tuple2<String, Integer>(s, n);
          }
        });

    Map<String, Integer> map = pairs.collectAsMap();

    twitter.map((String s) -> {
      try {
        JSONObject obj = (JSONObject) new JSONParser().parse(s);
        String twitternext = obj.get("tokens").toString();
        List<String> tokenlist = Arrays.asList(twitternext.split(" "));
        return map.get(tokenlist.get(0));
      } catch (ParseException e) {
        e.printStackTrace();
      }
      return 1;
    }).saveAsTextFile("aaaaaa");
    ;

  }

}
