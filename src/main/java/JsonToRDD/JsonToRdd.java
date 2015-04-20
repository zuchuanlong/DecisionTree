package JsonToRDD;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
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
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

public class JsonToRdd {

	public static void main(String args[]) {

		// saveJsonAsRdd();
		parseJson();

	}

	// save Json as JavaRDD
	public static void parseJson() {

		SparkConf sparkConf = new SparkConf().setAppName("SaveJsonAsRdd")
				.setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		String file = "Twitters.json";

		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			List<String> jsonData = Arrays.asList(br.readLine());
			JavaRDD<String> anotherPeopleRDD = sc.parallelize(jsonData);
			saveRDDAsHDFS(anotherPeopleRDD, "JsonAsRdd");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static void saveJsonAsRdd() {

		SparkConf sparkConf = new SparkConf().setAppName("SaveJsonAsRdd")
				.setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// sc is an existing JavaSparkContext.
		SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);

		// A JSON dataset is pointed to by path.
		// The path can be either a single text file or a directory storing text
		// files.
		String path = "people.json";
		// Create a DataFrame from the file(s) pointed to by path
		DataFrame people = sqlContext.jsonFile(path);

		// The inferred schema can be visualized using the printSchema() method.
		people.printSchema();
		// root
		// |-- age: integer (nullable = true)
		// |-- name: string (nullable = true)

		// people.show();

		// Register this DataFrame as a table.
		people.registerTempTable("people");

		// SQL statements can be run by using the sql methods provided by
		// sqlContext.
		DataFrame teenagers = sqlContext
				.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19");

		// teenagers.show();

		// Alternatively, a DataFrame can be created for a JSON dataset represented
		// by
		// an RDD[String] storing one JSON object per string.
		List<String> jsonData = Arrays
				.asList("{\"name\":\"Yin\",\"address\":{\"city\":\"Columbus\",\"state\":\"Ohio\"}}");
		JavaRDD<String> anotherPeopleRDD = sc.parallelize(jsonData);
		DataFrame anotherPeople = sqlContext.jsonRDD(anotherPeopleRDD);

		// anotherPeople.show();
		// anotherPeople.printSchema();
		// saveRDDAsHDFS(anotherPeopleRDD, "JsonAsRdd");

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
