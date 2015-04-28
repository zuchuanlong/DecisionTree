package DecisionTree;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class DecisionTree {

	SparkConf sparkConf = null;
	static JavaSparkContext sc = null;

	public static void main(String args[]) {

		new DecisionTree();
		createWeather();

	}

	public DecisionTree() {

		sparkConf = new SparkConf().setAppName("DataFrame").setMaster("local");
		sc = new JavaSparkContext(sparkConf);

	}

	public static void createWeather() {

		JavaRDD<String> weather = sc.textFile("weather.json");

		createWeatherHDFS(weather, "temperature", "weather/temperature");
		createWeatherHDFS(weather, "humidity", "weather/humidity");
		createWeatherHDFS(weather, "wind", "weather/wind");
		createWeatherHDFS(weather, "cloud", "weather/cloud");
		createWeatherHDFS(weather, "rain", "weather/rain");

		kMeans("weather/temperature/part-00000", 5, "weatherKMeans/temperature");
		kMeans("weather/humidity/part-00000", 5, "weatherKMeans/humidity");
		kMeans("weather/wind/part-00000", 5, "weatherKMeans/wind");
		kMeans("weather/cloud/part-00000", 5, "weatherKMeans/cloud");
		kMeans("weather/rain/part-00000", 5, "weatherKMeans/rain");

	}

	public static void createWeatherHDFS(JavaRDD<String> weather, String key,
			String savepath) {

		JavaRDD<String> attribute = weather.map(new Function<String, String>() {
			public String call(String s) {
				try {
					JSONObject obj = (JSONObject) new JSONParser().parse(s);
					return obj.get(key).toString();
				} catch (ParseException e) {
					e.printStackTrace();
					return null;
				}
			}
		});

		saveRDDAsHDFS(attribute, savepath);

	}

	public static void kMeans(String inputfile, int cluster, String savepath) {

		JavaRDD<String> data = sc.textFile(inputfile);
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

		int numClusters = cluster;
		int numIterations = 20;
		KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters,
				numIterations);

		Vector[] v = clusters.clusterCenters();

		saveRDDAsHDFS(
				clusters.predict(parsedData).map(new Function<Integer, String>() {
					public String call(Integer x) {
						return v[x].toString().replace("[", "").replace("]", "");
					}
				}), savepath);

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
