package Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class Matrix {

	public static void main(String args[]) {

		createMatrix();
		// System.out.println(createRow("gut wrench part leav job lose touch"));

	}

	public static void createMatrix() {

		SparkConf conf = new SparkConf().setAppName("CreateMatrix").setMaster(
				"local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> file = sc.textFile("Twitters.json");

		JavaRDD<String> matrix = file.map(new Function<String, String>() {
			public String call(String s) {
				try {
					JSONObject obj = (JSONObject) new JSONParser().parse(s);
					return createRow(obj.get("tokens").toString());
				} catch (ParseException e) {
					e.printStackTrace();
					return null;
				}
			}
		});

		saveRDDAsHDFS(matrix, "matrix");

	}

	public static String createRow(String tokens) {

		// String tokens = "gut wrench part leav job lose touch";
		List<String> tokenlist = Arrays.asList(tokens.split(" "));

		int tokenlength = tokenlist.size();

		String row = "";

		String file = "dictionary.txt";
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));

			String line = null;

			while ((line = br.readLine()) != null) {
				for (int n = 0; n < tokenlength; n++) {
					if (line.equals(tokenlist.get(n))) {
						row = row + " 1";
						System.out.println("1111111");
					} else {
						row = row + " 0";
						System.out.println("0000000");
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return row;

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
