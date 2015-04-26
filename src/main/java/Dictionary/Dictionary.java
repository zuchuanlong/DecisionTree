package Dictionary;

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
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import scala.Tuple2;

public class Dictionary {

	public static void main(String args[]) {

		createDictionary();

	}

	public static void createDictionary() {

		SparkConf conf = new SparkConf().setAppName("CreateDictionary").setMaster(
				"local");
		JavaSparkContext sc = new JavaSparkContext(conf);

		String path = "TwitterTest.json";

		JavaRDD<String> file = sc.textFile(path);
		JavaRDD<String> words = file.flatMap(new FlatMapFunction<String, String>() {
			public Iterable<String> call(String s) {
				try {
					JSONObject obj = (JSONObject) new JSONParser().parse(s);
					return Arrays.asList((obj.get("tokens").toString().split(" ")));
				} catch (ParseException e) {
					e.printStackTrace();
					return null;
				}
			}
		});

		// saveRDDAsHDFS(words, "dictionary");

		JavaPairRDD<String, Integer> pairs = words
				.mapToPair(new PairFunction<String, String, Integer>() {
					public Tuple2<String, Integer> call(String s) {
						return new Tuple2<String, Integer>(s, 1);
					}
				});
		JavaPairRDD<String, Integer> counts = pairs
				.reduceByKey(new Function2<Integer, Integer, Integer>() {
					public Integer call(Integer a, Integer b) {
						return a + b;
					}
				});

		// saveRDDAsHDFS(counts, "dictionary");

		saveRDDAsHDFS(counts.keys(), "dictionary");

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

}
