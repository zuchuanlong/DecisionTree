package DecisionTree;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class Test {

	public static void main(String args[]) {

		SparkConf conf = new SparkConf().setAppName("wordcount").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<String> file1 = sc
				.textFile("hdfs://localhost:9000/data/dictionary.txt");
		JavaRDD<String> file2 = sc
				.textFile("hdfs://localhost:9000/data/weather.json");

		file1.intersection(file2).saveAsTextFile("test");

	}

}
