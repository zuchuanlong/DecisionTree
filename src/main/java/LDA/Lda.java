package LDA;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.ivy.ant.IvyPublish.PublishArtifact;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import scala.Tuple2;

public class Lda {

	SparkConf conf = null;
	static JavaSparkContext sc = null;

	public static void main(String args[]) {

		new Lda();
		createDictionary();
		createMatrix();
		createLDA();

	}

	public Lda() {
		conf = new SparkConf().setAppName("LDA Example").setMaster("local");
		sc = new JavaSparkContext(conf);
	}

	public static void createDictionary() {

		String path = "Twitters.json";

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

	public static void createMatrix() {

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

	public static void createLDA() {

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
		// JavaLDAExample.saveRDDAsHDFS(parsedData, "parsedData");

		// Index documents with unique IDs
		JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(parsedData
				.zipWithIndex().map(
						new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
							public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id) {
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
