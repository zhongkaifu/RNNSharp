using AdvUtils;
using RNNSharp;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace RNNSharpConsole
{
    /// <summary>
    ///     RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
    /// </summary>
    internal class Program
    {
        private static string testFilePath = "";
        private static string outputFilePath = "";
        private static string tagFilePath = "";
        private static string configFilePath = "";
        private static string trainFilePath = "";
        private static string validFilePath = "";
        private static int maxIter = 20;
        private static long savestep;
        private static float alpha = 0.1f;
        private static int nBest = 1;
        private static int iVQ;
        private static float gradientCutoff = 15.0f;
        private static bool constAlpha;

        private static void UsageTitle()
        {
            Console.WriteLine("Deep Recurrent Neural Network Toolkit v2.2.1 by Zhongkai Fu (fuzhongkai@gmail.com)");
        }

        private static void Usage()
        {
            UsageTitle();
            Console.WriteLine("RNNSharpConsole.exe <parameters>");
            Console.WriteLine(" -mode <string>: train/test");
        }

        private static void UsageTrain()
        {
            UsageTitle();
            Console.WriteLine("RNNSharpConsole.exe -mode train <parameters>");
            Console.WriteLine("Parameters for Deep-RNN training");

            Console.WriteLine(" -trainfile <string>");
            Console.WriteLine("\tTraining corpus file");

            Console.WriteLine(" -validfile <string>");
            Console.WriteLine("\tValidated corpus file for training");

            Console.WriteLine(" -cfgfile <string>");
            Console.WriteLine("\tConfiguration file");

            Console.WriteLine(" -tagfile <string>");
            Console.WriteLine("\tSupported output tagid-name list file");

            Console.WriteLine(" -alpha <float>");
            Console.WriteLine("\tInitializing learning rate. Default is 0.1");

            Console.WriteLine(" -constalpha <boolean>");
            Console.WriteLine("\tUse const learning rat. Default is false");

            Console.WriteLine(" -maxiter <int>");
            Console.WriteLine("\tMaximum iteration for training, 0 is unlimited. Default is 20");

            Console.WriteLine(" -savestep <int>");
            Console.WriteLine("\tSave temporary model after every <int> sentences. Default is 0");

            Console.WriteLine(" -vq <int>");
            Console.WriteLine("\tModel vector quantization, 0 is disable, 1 is enable. Default is 0");

            Console.WriteLine(" -grad <float>");
            Console.WriteLine("\tGradient cut-off. Default is 15.0f");

            Console.WriteLine();
            Console.WriteLine(
                "Example: RNNSharpConsole.exe -mode train -trainfile train.txt -validfile valid.txt -cfgfile config.txt -tagfile tags.txt -alpha 0.1 -maxiter 20 -savestep 200K -vq 0 -grad 15.0");
        }

        private static void UsageTest()
        {
            UsageTitle();
            Console.WriteLine("RNNSharpConsole.exe -mode test <parameters>");
            Console.WriteLine("Parameters to predict tags from given corpus by using trained model");
            Console.WriteLine(" -testfile <string>");
            Console.WriteLine("\tInput file for testing");

            Console.WriteLine(" -tagfile <string>");
            Console.WriteLine("\tSupported output tagid-name list file");

            Console.WriteLine(" -cfgfile <string>");
            Console.WriteLine("\tConfiguration file");

            Console.WriteLine(" -outfile <string>");
            Console.WriteLine("\tResult output file");

            Console.WriteLine(" -nbest <int>");
            Console.WriteLine("\tGet N-best result. Default is 1");

            Console.WriteLine();
            Console.WriteLine(
                "Example: RNNSharpConsole.exe -mode test -testfile test.txt -modelfile model.bin -tagfile tags.txt -ftrfile features.txt -outfile result.txt -nbest 3");
        }

        private static void InitParameters(string[] args)
        {
            int i;
            if ((i = ArgPos("-trainfile", args)) >= 0) trainFilePath = args[i + 1];
            if ((i = ArgPos("-validfile", args)) >= 0) validFilePath = args[i + 1];
            if ((i = ArgPos("-testfile", args)) >= 0) testFilePath = args[i + 1];
            if ((i = ArgPos("-outfile", args)) >= 0) outputFilePath = args[i + 1];
            if ((i = ArgPos("-tagfile", args)) >= 0) tagFilePath = args[i + 1];
            if ((i = ArgPos("-cfgfile", args)) >= 0) configFilePath = args[i + 1];
            if ((i = ArgPos("-maxiter", args)) >= 0) maxIter = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
            if ((i = ArgPos("-alpha", args)) >= 0) alpha = float.Parse(args[i + 1], CultureInfo.InvariantCulture);
            if ((i = ArgPos("-nbest", args)) >= 0) nBest = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
            if ((i = ArgPos("-vq", args)) >= 0) iVQ = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
            if ((i = ArgPos("-grad", args)) >= 0)
                gradientCutoff = float.Parse(args[i + 1], CultureInfo.InvariantCulture);
            if ((i = ArgPos("-constalpha", args)) >= 0) constAlpha = bool.Parse(args[i + 1]);

            if ((i = ArgPos("-savestep", args)) >= 0)
            {
                var str = args[i + 1].ToLower();
                if (str.EndsWith("k"))
                {
                    savestep = long.Parse(str.Substring(0, str.Length - 1), CultureInfo.InvariantCulture) * 1024;
                }
                else if (str.EndsWith("m"))
                {
                    savestep = long.Parse(str.Substring(0, str.Length - 1), CultureInfo.InvariantCulture) * 1024 * 1024;
                }
                else if (str.EndsWith("g"))
                {
                    savestep = long.Parse(str.Substring(0, str.Length - 1), CultureInfo.InvariantCulture) * 1024 * 1024 * 1024;
                }
                else
                {
                    savestep = long.Parse(str, CultureInfo.InvariantCulture);
                }
            }
        }

        //Parse parameters in command line
        private static int ArgPos(string str, string[] args)
        {
            str = str.ToLower();
            for (var a = 0; a < args.Length; a++)
            {
                if (str == args[a].ToLower())
                {
                    if (a == args.Length - 1)
                    {
                        Logger.WriteLine("Argument missing for {0}", str);
                        return -1;
                    }
                    return a;
                }
            }
            return -1;
        }

        //Check if the corpus is validated and get the number of record in total
        private static void CheckCorpus(string strFileName)
        {
            var sr = new StreamReader(strFileName);
            var RecordCount = 0;

            var tokenList = new List<string>();
            while (true)
            {
                var strLine = sr.ReadLine();
                if (strLine == null)
                {
                    break;
                }

                strLine = strLine.Trim();
                if (strLine.Length == 0)
                {
                    RecordCount++;
                }
            }

            Logger.WriteLine("Record set size in {0}: {1}", strFileName, RecordCount);
            sr.Close();
        }

        private static void LoadSeq2SeqDataSet(string strFileName, Config featurizer, DataSet<SequencePair> dataSet)
        {
            Logger.WriteLine("Loading data set for seq2seq2 training...");
            var sr = new StreamReader(strFileName);
            var RecordCount = 0;
            while (true)
            {
                var sentPair = new SentencePair
                {
                    srcSentence = new Sentence(ReadRecord(sr)),
                    tgtSentence = new Sentence(ReadRecord(sr), false)
                };

                //Extract features from it and convert it into sequence

                if (sentPair.srcSentence.TokensList.Count <= 2 || sentPair.tgtSentence.TokensList.Count <= 0)
                {
                    //No more record, it only contain <s> and </s>
                    break;
                }

                var seq = featurizer.ExtractFeatures(sentPair);
                if (seq.tgtSequence.SetLabel(sentPair.tgtSentence, featurizer.TagSet))
                {
                    dataSet.SequenceList.Add(seq);

                    //Show state at every 1000 record
                    RecordCount++;
                    if (RecordCount % 10000 == 0)
                    {
                        Logger.WriteLine("{0}...", RecordCount);
                    }
                }
            }

            sr.Close();
        }

        private static void LoadDataset(string strFileName, Config featurizer, DataSet<Sequence> dataSet)
        {
            Logger.WriteLine("Loading data set...");
            CheckCorpus(strFileName);

            var sr = new StreamReader(strFileName);
            var RecordCount = 0;

            while (true)
            {
                try
                {
                    //Extract features from it and convert it into sequence
                    var sent = new Sentence(ReadRecord(sr));
                    if (sent.TokensList.Count <= 2)
                    {
                        //No more record, it only contain <s> and </s>
                        break;
                    }

                    var seq = featurizer.BuildSequence(sent);

                    //Set label for the sequence
                    if (seq.SetLabel(sent, featurizer.TagSet))
                    {
                        //Add the sequence into data set
                        dataSet.SequenceList.Add(seq);

                        //Show state at every 1000 record
                        RecordCount++;
                        if (RecordCount % 10000 == 0)
                        {
                            Logger.WriteLine("{0}...", RecordCount);
                        }
                    }
                }
                catch (Exception err)
                {
                    Logger.WriteLine("Fail to parse corpus: '{0}'", err.Message);
                }
            }

            sr.Close();
        }

        private static void Main(string[] args)
        {
            int i;

            //Initialize all given parameters
            InitParameters(args);

            if ((i = ArgPos("-mode", args)) >= 0)
            {
                var strMode = args[i + 1].ToLower();
                switch (strMode)
                {
                    case "train":
                        Train();
                        break;

                    case "test":
                        Test();
                        break;
                }
            }
            else
            {
                Logger.WriteLine(Logger.Level.err, "Running mode is required.");
                Usage();
            }
        }

        private static List<string[]> ReadRecord(StreamReader sr)
        {
            var record = new List<string[]>();
            string strLine;

            //Read each line from file
            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.Length == 0)
                {
                    //If read a empty line, it should be the end of current record
                    return record;
                }

                record.Add(strLine.Split('\t'));
            }

            return record;
        }

        private static void Test()
        {
            if (string.IsNullOrEmpty(tagFilePath))
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The tag mapping file {0} isn't specified.", tagFilePath);
                UsageTest();
                return;
            }

            //Load tag name
            Logger.WriteLine($"Loading tag file '{tagFilePath}'");
            var tagSet = new TagSet(tagFilePath);

            if (string.IsNullOrEmpty(configFilePath))
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The configuration file {0} isn't specified.", configFilePath);
                UsageTest();
                return;
            }

            if (outputFilePath.Length == 0)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The output file name should not be empty.");
                UsageTest();
                return;
            }

            //Create feature extractors and load word embedding data from file
            Logger.WriteLine($"Initializing config file = '{configFilePath}'");
            var config = new Config(configFilePath, tagSet);
            config.ShowFeatureSize();

            //Create instance for decoder
            Logger.WriteLine($"Loading model from {config.ModelFilePath} and creating decoder instance...");
            var decoder = new RNNDecoder(config);

            if (File.Exists(testFilePath) == false)
            {
                Logger.WriteLine(Logger.Level.err, $"FAILED: The test corpus {testFilePath} doesn't exist.");
                UsageTest();
                return;
            }

            var sr = new StreamReader(testFilePath);
            var sw = new StreamWriter(outputFilePath);

            while (true)
            {
                var sent = new Sentence(ReadRecord(sr));
                if (sent.TokensList.Count <= 2)
                {
                    //No more record, it only contains <s> and </s>
                    break;
                }

                if (nBest == 1)
                {
                    //Output decoded result
                    if (decoder.ModelType == MODELTYPE.SeqLabel)
                    {
                        //Append the decoded result into the end of feature set of each token
                        var output = decoder.Process(sent);
                        var sb = new StringBuilder();
                        for (var i = 0; i < sent.TokensList.Count; i++)
                        {
                            var tokens = string.Join("\t", sent.TokensList[i]);
                            sb.Append(tokens);
                            sb.Append("\t");
                            sb.Append(tagSet.GetTagName(output[i]));
                            sb.AppendLine();
                        }

                        sw.WriteLine(sb.ToString());
                    }
                    else
                    {
                        //Print out source sentence at first, and then generated result sentence
                        var output = decoder.ProcessSeq2Seq(sent);
                        var sb = new StringBuilder();
                        for (var i = 0; i < sent.TokensList.Count; i++)
                        {
                            var tokens = string.Join("\t", sent.TokensList[i]);
                            sb.AppendLine(tokens);
                        }
                        sw.WriteLine(sb.ToString());
                        sw.WriteLine();

                        sb.Clear();
                        for (var i = 0; i < output.Length; i++)
                        {
                            var token = tagSet.GetTagName(output[i]);
                            sb.AppendLine(token);
                        }
                        sw.WriteLine(sb.ToString());
                        sw.WriteLine();
                    }
                }
                else
                {
                    var output = decoder.ProcessNBest(sent, nBest);
                    var sb = new StringBuilder();
                    for (var i = 0; i < nBest; i++)
                    {
                        for (var j = 0; j < sent.TokensList.Count; j++)
                        {
                            var tokens = string.Join("\t", sent.TokensList[i]);
                            sb.Append(tokens);
                            sb.Append("\t");
                            sb.Append(tagSet.GetTagName(output[i][j]));
                            sb.AppendLine();
                        }
                        sb.AppendLine();
                    }

                    sw.WriteLine(sb.ToString());
                }
            }

            sr.Close();
            sw.Close();
        }

        public static LayerType ParseLayerType(string layerType)
        {
            foreach (
                var type in
                    Enum.GetValues(typeof(LayerType))
                        .Cast<LayerType>()
                        .Where(type => layerType.Equals(type.ToString(), StringComparison.InvariantCultureIgnoreCase)))
            {
                return type;
            }

            throw new ArgumentException($"Invalidated layer type: {layerType}");
        }

        private static void Train()
        {
            Logger.LogFile = "RNNSharpConsole.log";

            if (File.Exists(tagFilePath) == false)
            {
                Logger.WriteLine(Logger.Level.err, $"FAILED: The tag mapping file {tagFilePath} doesn't exist.");
                UsageTrain();
                return;
            }

            //Load tag id and its name from file
            var tagSet = new TagSet(tagFilePath);

            //Create configuration instance and set parameters
            var RNNConfig = new ModelSetting
            {
                VQ = iVQ,
                MaxIteration = maxIter,
                SaveStep = savestep,
                LearningRate = alpha,
                GradientCutoff = gradientCutoff,
                IsConstAlpha = constAlpha
            };

            //Dump RNN setting on console
            RNNConfig.DumpSetting();

            if (File.Exists(configFilePath) == false)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The feature configuration file {0} doesn't exist.",
                    configFilePath);
                UsageTrain();
                return;
            }
            //Create feature extractors and load word embedding data from file
            var config = new Config(configFilePath, tagSet);
            config.ShowFeatureSize();

            if (File.Exists(trainFilePath) == false)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The training corpus doesn't exist.");
                UsageTrain();
                return;
            }

            if (config.ModelType == MODELTYPE.SeqLabel)
            {
                //Create RNN encoder and save necessary parameters
                var encoder = new RNNEncoder<Sequence>(RNNConfig, config)
                {
                    TrainingSet = new DataSet<Sequence>(tagSet.GetSize())
                };

                //LoadFeatureConfig training corpus and extract feature set
                LoadDataset(trainFilePath, config, encoder.TrainingSet);

                if (string.IsNullOrEmpty(validFilePath) == false)
                {
                    //LoadFeatureConfig validated corpus and extract feature set
                    Logger.WriteLine("Loading validated corpus from {0}", validFilePath);
                    encoder.ValidationSet = new DataSet<Sequence>(tagSet.GetSize());
                    LoadDataset(validFilePath, config, encoder.ValidationSet);
                }
                else
                {
                    Logger.WriteLine("Validated corpus isn't specified.");
                    encoder.ValidationSet = null;
                }

                if (encoder.IsCRFTraining)
                {
                    Logger.WriteLine("Initialize output tag bigram transition probability...");
                    //Build tag bigram transition matrix
                    encoder.TrainingSet.BuildLabelBigramTransition();
                }

                //Start to train the model
                encoder.Train();
            }
            else
            {
                //Create RNN encoder and save necessary parameters
                var encoder = new RNNEncoder<SequencePair>(RNNConfig, config)
                {
                    TrainingSet = new DataSet<SequencePair>(tagSet.GetSize())
                };

                //LoadFeatureConfig training corpus and extract feature set

                LoadSeq2SeqDataSet(trainFilePath, config, encoder.TrainingSet);

                if (string.IsNullOrEmpty(validFilePath) == false)
                {
                    //LoadFeatureConfig validated corpus and extract feature set
                    Logger.WriteLine("Loading validated corpus from {0}", validFilePath);
                    encoder.ValidationSet = new DataSet<SequencePair>(tagSet.GetSize());
                    LoadSeq2SeqDataSet(validFilePath, config, encoder.ValidationSet);
                }
                else
                {
                    Logger.WriteLine("Validated corpus isn't specified.");
                    encoder.ValidationSet = null;
                }

                if (encoder.IsCRFTraining)
                {
                    Logger.WriteLine("Initialize output tag bigram transition probability...");
                    //Build tag bigram transition matrix
                    encoder.TrainingSet.BuildLabelBigramTransition();
                }

                //Start to train the model
                encoder.Train();
            }
        }
    }
}