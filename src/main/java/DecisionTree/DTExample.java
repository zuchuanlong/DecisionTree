package DecisionTree;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;

public class DTExample {

	public static void main(String[] args) {

		decisionTree();

	}

	public static void decisionTree() {

		SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree")
				.setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Load and parse the data file.
		String datapath = "DT_data.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath)
				.toJavaRDD();

		// Split the data into training and test sets (30% held out for testing)
		JavaRDD<LabeledPoint>[] splits = data
				.randomSplit(new double[] { 0.7, 0.3 });
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];

		// Set parameters.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 5;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		String impurity = "gini";
		Integer maxDepth = 5;
		Integer maxBins = 32;

		// Train a DecisionTree model for classification.
		final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData,
				numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel = testData
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p
								.label());
					}
				});

		// System.out.println(predictionAndLabel);

		Double testErr = 1.0
				* predictionAndLabel.filter(
						new Function<Tuple2<Double, Double>, Boolean>() {
							public Boolean call(Tuple2<Double, Double> pl) {
								return !pl._1().equals(pl._2());
							}
						}).count() / testData.count();
		System.out.println("Test Error: " + testErr);
		System.out.println("Learned classification tree model:\n"
				+ model.toDebugString());

		// Save and load model
		// model.save(sc.sc(), "myModelPath");
		// DecisionTreeModel sameModel = DecisionTreeModel.load(sc.sc(),
		// "myModelPath");

		// saveRDDAsHDFS(predictionAndLabel, "predictionAndLabel");
		saveRDDAsHDFS(data, "data");

	}

	public static void saveRDDAsHDFS(JavaRDD<LabeledPoint> tweets, String fileOut) {
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
