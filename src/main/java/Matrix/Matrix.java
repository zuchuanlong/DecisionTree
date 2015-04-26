package Matrix;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class Matrix {

	public static void main(String args[]) {

		createMatrix();

	}

	public static void createMatrix() {

		SparkConf conf = new SparkConf().setAppName("CreateMatrix").setMaster(
				"local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> twitter = sc.textFile("Twitters.json");
		JavaRDD<String> dictionary = sc.textFile("dictionary/part-00000");

		Iterator twitterIte = twitter.toLocalIterator();
		Iterator dictionaryIte = null;

		String matrixIte = "";
		String twitternext = null;
		String dictionarynext = null;
		List<String> tokenlist = null;
		int tokenlength = 0;

		int number = 0;

		JSONObject obj = new JSONObject();
		JSONParser parse = new JSONParser();

		while (twitterIte.hasNext()) {

			twitternext = twitterIte.next().toString();
			try {
				obj = (JSONObject) parse.parse(twitternext);
			} catch (ParseException e) {
				e.printStackTrace();
			}
			twitternext = obj.get("tokens").toString();
			tokenlist = Arrays.asList(twitternext.split(" "));
			tokenlength = tokenlist.size();

			dictionaryIte = dictionary.toLocalIterator();

			while (dictionaryIte.hasNext()) {
				dictionarynext = dictionaryIte.next().toString();
				for (int n = 0; n < tokenlength; n++) {
					if (dictionarynext.equals(tokenlist.get(n))) {
						number++;
						break;
					}
				}
				matrixIte = matrixIte + " " + String.valueOf(number);
				number = 0;
			}
			matrixIte = matrixIte + "\n";
		}

		saveRDDAsHDFS(sc.parallelize(Arrays.asList(matrixIte)), "matrix");

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
