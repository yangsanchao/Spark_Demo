/**
  * Created by ysc on 17/5/18.
  */
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

/*
* 以逻辑斯蒂回归为例，构建一个典型的机器学习过程，来具体介绍一下工作流是如何应用的。
* 目的是查找出所有包含"spark"的句子，即将包含"spark"的句子的标签设为1，没有"spark"的句子的标签设为0。
* */

object Pipelines_Test {

  def main(args: Array[String]): Unit = {
    mlPipelines
  }

  def mlPipelines: Unit = {

    val spark = SparkSession.builder().
      master("local").
      appName("my App Name").
      getOrCreate()

    val training = spark.createDataFrame(Seq(
             (0L, "a b c d e spark", 1.0),
             (1L, "b d", 0.0),
             (2L, "spark f g h", 1.0),
             (3L, "hadoop mapreduce", 0.0)
             )).toDF("id", "text", "label")

    val tokenizer = new Tokenizer().
             setInputCol("text").
             setOutputCol("words")

    val hashingTF = new HashingTF().
             setNumFeatures(1000).
             setInputCol(tokenizer.getOutputCol).
             setOutputCol("features")

    val lr = new LogisticRegression().
             setMaxIter(10).
             setRegParam(0.01)


    val pipeline = new Pipeline().
             setStages(Array(tokenizer, hashingTF, lr))

    val model = pipeline.fit(training)

    val test = spark.createDataFrame(Seq(
             (4L, "spark i j k"),
             (5L, "l m n"),
             (6L, "spark a"),
             (7L, "apache hadoop")
             )).toDF("id", "text")

    model.transform(test).
             select("id", "text", "probability", "prediction").
             collect().
             foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
               println(s"($id, $text) --> prob=$prob, prediction=$prediction")
             }
  }

}
