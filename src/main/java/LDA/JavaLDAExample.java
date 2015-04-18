package LDA;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import scala.Tuple2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.apache.spark.SparkConf;

public class JavaLDAExample {

	public static void main(String[] args) {

		javaLDA();

	}

	public static void javaLDA() {

		SparkConf conf = new SparkConf().setAppName("LDA Example").setMaster(
				"local");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load and parse the data
		String path = "/Users/chuanlongzu/Downloads/spark-1.3.0/data/mllib/sample_lda_data.txt";
		JavaRDD<String> data = sc.textFile(path);
		JavaRDD<Vector> parsedData = data.map(new Function<String, Vector>() {
			public Vector call(String s) {
				String[] sarray = s.trim().split(" ");
				double[] values = new double[sarray.length];
				for (int i = 0; i < sarray.length; i++)
					values[i] = Double.parseDouble(sarray[i]);
				return Vectors.dense(values);
			}
		});
		// JavaLDAExample.saveRDDAsHDFS(parsedData, "parsedData");

		// Index documents with unique IDs
		JavaPairRDD<Long, Vector> corpus = JavaPairRDD
				.fromJavaRDD(parsedData
						.zipWithIndex()
						.map(new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
							public Tuple2<Long, Vector> call(
									Tuple2<Vector, Long> doc_id) {
								return doc_id.swap();
							}
						}));
		corpus.cache();
		// JavaLDAExample.saveRDDAsHDFS(corpus, "corpus");

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

		RDD<Tuple2<Object, Vector>> td = ldaModel.topicDistributions();
		JavaLDAExample.saveRDDAsHDFS(td, "td");

	}

	protected static void saveRDDAsHDFS(RDD<Tuple2<Object, Vector>> tweets,
			String fileOut) {
		try {
			URI fileOutURI = new URI(fileOut);
			URI hdfsURI = new URI(fileOutURI.getScheme(), null,
					fileOutURI.getHost(), fileOutURI.getPort(), null, null,
					null);
			Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
			FileSystem hdfs = org.apache.hadoop.fs.FileSystem.get(hdfsURI,
					hadoopConf);
			System.out.print(hdfsURI.toString()); // XXX
			System.out.print(fileOutURI.toString()); // XXX
			hdfs.delete(new org.apache.hadoop.fs.Path(fileOut), true);
			tweets.saveAsTextFile(fileOut);
		} catch (URISyntaxException | IOException e) {
			Logger.getRootLogger().error(e);
		}
	}
}