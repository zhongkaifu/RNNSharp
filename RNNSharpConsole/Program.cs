using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using RNNSharp;
using AdvUtils;

namespace RNNSharpConsole
{
    class Program
    {
        static string strModelFile = "";
        static string strTestFile = "";
        static string strOutputFile = "";
        static string strTagFile = "";
        static string strFeatureConfigFile = "";
        static string strTrainFile = "";
        static string strValidFile = "";
        static int maxIter = 20;
        static int layersize = 200;
        static int iCRF = 0;
        static long savestep = 0;
        static double alpha = 0.1;
        static double dropout = 0;
        static int bptt = 4;
        static int modelType = 0;
        static int nBest = 1;
        static int iDir = 0;

        static void UsageTitle()
        {
            Console.WriteLine("Recurrent Neural Network Toolkit v1.1 by Zhongkai Fu (fuzhongkai@gmail.com)");
        }

        static void Usage()
        {
            UsageTitle();
            Console.WriteLine("RNNSharpConsole.exe <parameters>");
            Console.WriteLine(" -mode <string>: train/test");
        }

        static void UsageTrain()
        {
            UsageTitle();
            Console.WriteLine("RNNSharpConsole.exe -mode train <parameters>");
            Console.WriteLine("Parameters for RNN training");

            Console.WriteLine(" -trainfile <string>");
            Console.WriteLine("\tTraining corpus file");

            Console.WriteLine(" -validfile <string>");
            Console.WriteLine("\tValidated corpus file for training");

            Console.WriteLine(" -modelfile <string>");
            Console.WriteLine("\tEncoded model file");

            Console.WriteLine(" -modeltype <int>");
            Console.WriteLine("\tModel type: 0 - BPTT-RNN, 1 - LSTM-RNN, default is 0");

            Console.WriteLine(" -dir <int>");
            Console.WriteLine("\tRecurrent direction: 0 - Forward RNN, 1 - Bi-directional RNN, default is 0");

            Console.WriteLine(" -ftrfile <string>");
            Console.WriteLine("\tFeature configuration file");

            Console.WriteLine(" -tagfile <string>");
            Console.WriteLine("\tSupported output tagid-name list file");

            Console.WriteLine(" -alpha <float>");
            Console.WriteLine("\tInitializing learning rate, default is 0.1");

            Console.WriteLine(" -dropout <float>");
            Console.WriteLine("\tDropout ratio [0, 1.0), default is 0");

            Console.WriteLine(" -layersize <int>");
            Console.WriteLine("\tHidden layer size, default is 200");

            Console.WriteLine(" -bptt <int>");
            Console.WriteLine("\tStep for back-propagation through time for BPTT-RNN. default is 4");

            Console.WriteLine(" -crf <int>");
            Console.WriteLine("\tEnable CRF model at output, 0 is disable, 1 is enable. default is 0");

            Console.WriteLine(" -maxiter <int>");
            Console.WriteLine("\tMaximum iteration for training, 0 is unlimited. default is 20");

            Console.WriteLine(" -savestep <int>");
            Console.WriteLine("\tSave temporary model after every <int> sentences, default is 0");

            Console.WriteLine();
            Console.WriteLine("Example: RNNSharpConsole.exe -mode train -trainfile train.txt -validfile valid.txt -modelfile model.bin -ftrfile features.txt -tagfile tags.txt -modeltype 0 -layersize 200 -alpha 0.1 -crf 1 -maxiter 20 -savestep 200K -dir 0");

        }

        static void UsageTest()
        {
            UsageTitle();
            Console.WriteLine("RNNSharpConsole.exe -mode test <parameters>");
            Console.WriteLine("Parameters to predict tags from given corpus by using trained model");
            Console.WriteLine(" -testfile <string>");
            Console.WriteLine("\tTraining corpus file");

            Console.WriteLine(" -modelfile <string>");
            Console.WriteLine("\tEncoded model file");

            Console.WriteLine(" -tagfile <string>");
            Console.WriteLine("\tSupported output tagid-name list file");

            Console.WriteLine(" -ftrfile <string>");
            Console.WriteLine("\tFeature configuration file");

            Console.WriteLine(" -outfile <string>");
            Console.WriteLine("\tResult output file");

            Console.WriteLine(" -nbest <int>");
            Console.WriteLine("\tGet N-best result. Default is 1");

            Console.WriteLine();
            Console.WriteLine("Example: RNNSharpConsole.exe -mode test -testfile test.txt -modelfile model.bin -tagfile tags.txt -ftrfile features.txt -outfile result.txt -nbest 3");
        }

        static void InitParameters(string[] args)
        {
            int i;
            if ((i = ArgPos("-modelfile", args)) >= 0) strModelFile = args[i + 1];
            if ((i = ArgPos("-trainfile", args)) >= 0) strTrainFile = args[i + 1];
            if ((i = ArgPos("-validfile", args)) >= 0) strValidFile = args[i + 1];
            if ((i = ArgPos("-testfile", args)) >= 0) strTestFile = args[i + 1];
            if ((i = ArgPos("-outfile", args)) >= 0) strOutputFile = args[i + 1];
            if ((i = ArgPos("-tagfile", args)) >= 0) strTagFile = args[i + 1];
            if ((i = ArgPos("-ftrfile", args)) >= 0) strFeatureConfigFile = args[i + 1];

            if ((i = ArgPos("-layersize", args)) >= 0) layersize = int.Parse(args[i + 1]);
            if ((i = ArgPos("-modeltype", args)) >= 0) modelType = int.Parse(args[i + 1]);
            if ((i = ArgPos("-crf", args)) >= 0) iCRF = int.Parse(args[i + 1]);
            if ((i = ArgPos("-maxiter", args)) >= 0) maxIter = int.Parse(args[i + 1]);
            if ((i = ArgPos("-alpha", args)) >= 0) alpha = double.Parse(args[i + 1]);
            if ((i = ArgPos("-dropout", args)) >= 0) dropout = double.Parse(args[i + 1]);
            if ((i = ArgPos("-bptt", args)) >= 0) bptt = int.Parse(args[i + 1]);
            if ((i = ArgPos("-nbest", args)) >= 0) nBest = int.Parse(args[i + 1]);
            if ((i = ArgPos("-dir", args)) >= 0) iDir = int.Parse(args[i + 1]);

            if ((i = ArgPos("-savestep", args)) >= 0)
            {
                string str = args[i + 1].ToLower();
                if (str.EndsWith("k") == true)
                {
                    savestep = long.Parse(str.Substring(0, str.Length - 1)) * 1024;
                }
                else if (str.EndsWith("m") == true)
                {
                    savestep = long.Parse(str.Substring(0, str.Length - 1)) * 1024 * 1024;
                }
                else if (str.EndsWith("g") == true)
                {
                    savestep = long.Parse(str.Substring(0, str.Length - 1)) * 1024 * 1024 * 1024;
                }
                else
                {
                    savestep = long.Parse(str);
                }
            }

        }

        //Parse parameters in command line
        static int ArgPos(string str, string[] args)
        {
            str = str.ToLower();
            for (int a = 0; a < args.Length; a++)
            {
                if (str == args[a].ToLower())
                {
                    if (a == args.Length - 1)
                    {
                        Logger.WriteLine(Logger.Level.info, "Argument missing for {0}", str);
                        return -1;
                    }
                    return a;
                }
            }
            return -1;
        }

        //Check if the corpus is validated and get the number of record in total
        static void CheckCorpus(string strFileName)
        {
            StreamReader sr = new StreamReader(strFileName);
            string strLine = null;
            int RecordCount = 0;

            List<string> tokenList = new List<string>();
            while (true)
            {
                strLine = sr.ReadLine();
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

            Logger.WriteLine(Logger.Level.info, "Record set size in {0}: {1}", strFileName, RecordCount);
            sr.Close();

        }

        static void LoadDataset(string strFileName, Featurizer featurizer, DataSet dataSet)
        {
            CheckCorpus(strFileName);

            StreamReader sr = new StreamReader(strFileName);
            int RecordCount = 0;

            while (true)
            {
                List<string> tokenList = ReadRecord(sr);
                if (tokenList.Count == 0)
                {
                    //No more record
                    break;
                }

                //Extract features from it and convert it into sequence
                Sentence sent = new Sentence();
                sent.SetFeatures(tokenList);
                Sequence seq = featurizer.ExtractFeatures(sent);
                
                //Set label for the sequence
                if (seq.SetLabel(sent, featurizer.GetTagSet()) == false)
                {
                    Logger.WriteLine(Logger.Level.info, "Error: Invalidated record.");
                    sent.DumpFeatures();
                    continue;
                }

                //Add the sequence into data set
                dataSet.Add(seq);

                //Show state at every 1000 record
                RecordCount++;
                if (RecordCount % 10000 == 0)
                {
                    Logger.WriteLine(Logger.Level.info, "{0}...", RecordCount);
                }
            }

            sr.Close();

        }


        static void Main(string[] args)
        {
            string strMode = "train";
            int i;

            //Initialize all given parameters
            InitParameters(args);

            if ((i = ArgPos("-mode", args)) >= 0)
            {
                strMode = args[i + 1].ToLower();
                if (strMode == "train")
                {
                    Train();
                }
                else if (strMode == "test")
                {
                    Test();
                }
            }
            else
            {
                Logger.WriteLine(Logger.Level.err, "Running mode is required.");
                Usage();
            }
        }

        private static List<string> ReadRecord(StreamReader sr)
        {
            List<string> record = new List<string>();
            string strLine = null;

            //Read each line from file
            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.Length == 0)
                {
                    //If read a empty line, it should be the end of current record
                    return record;
                }

                record.Add(strLine);
            }

            return record;
        }

        private static void Test()
        {
            if (String.IsNullOrEmpty(strTagFile) == true)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The tag mapping file {0} isn't specified.", strTagFile);
                UsageTest();
                return;
            }

            //Load tag id and its name from file
            TagSet tagSet = new TagSet(strTagFile);

            if (String.IsNullOrEmpty(strModelFile) == true)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The model file {0} isn't specified.", strModelFile);
                UsageTest();
                return;
            }

            if (String.IsNullOrEmpty(strFeatureConfigFile) == true)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The feature configuration file {0} isn't specified.", strFeatureConfigFile);
                UsageTest();
                return;
            }

            if (strOutputFile.Length == 0)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The output file name should not be empty.");
                UsageTest();
                return;
            }

            //Create feature extractors and load word embedding data from file
            Featurizer featurizer = new Featurizer(strFeatureConfigFile, tagSet);
            featurizer.ShowFeatureSize();

            //Create an instance for the model
           // Model model = new Model(strModelFile);

            //Create instance for decoder
            RNNSharp.RNNDecoder decoder = new RNNSharp.RNNDecoder(strModelFile, featurizer);

            if (File.Exists(strTestFile) == false)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The test corpus {0} isn't existed.", strTestFile);
                UsageTest();
                return;
            }

            StreamReader sr = new StreamReader(strTestFile);
            StreamWriter sw = new StreamWriter(strOutputFile);

            while (true)
            {
                List<string> tokenList = ReadRecord(sr);
                if (tokenList.Count == 0)
                {
                    //No more record
                    break;
                }

                Sentence sent = new Sentence();
                sent.SetFeatures(tokenList);

                if (nBest == 1)
                {
                    int[] output = decoder.Process(sent);
                    //Output decoded result
                    //Append the decoded result into the end of feature set of each token
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < tokenList.Count; i++)
                    {
                        sb.Append(tokenList[i]);
                        sb.Append("\t");
                        sb.Append(tagSet.GetTagName(output[i]));
                        sb.AppendLine();
                    }

                    sw.WriteLine(sb.ToString());
                }
                else
                {
                    int[][] output = decoder.ProcessNBest(sent, nBest);
                    if (output == null)
                    {
                        Logger.WriteLine(Logger.Level.err, "FAILED: decode failed. Dump current sentence...");
                        sent.DumpFeatures();
                        return;
                    }

                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < nBest; i++)
                    {
                        for (int j = 0; j < tokenList.Count; j++)
                        {
                            sb.Append(tokenList[j]);
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

        private static void Train()
        {
            Logger.LogFile = "RNNSharpConsole.log";

            if (File.Exists(strTagFile) == false)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The tag mapping file {0} isn't existed.", strTagFile);
                UsageTrain();
                return;
            }

            //Load tag id and its name from file
            TagSet tagSet = new TagSet(strTagFile);

            //Create configuration instance and set parameters
            ModelSetting RNNConfig = new ModelSetting();
            RNNConfig.SetModelFile(strModelFile);
            RNNConfig.SetNumHidden(layersize);
            RNNConfig.SetCRFTraining((iCRF == 1) ? true : false);
            RNNConfig.SetDir(iDir);
            RNNConfig.SetModelType(modelType);
            RNNConfig.SetMaxIteration(maxIter);
            RNNConfig.SetSaveStep(savestep);
            RNNConfig.SetLearningRate(alpha);
            RNNConfig.SetDropout(dropout);
            RNNConfig.SetBptt(bptt);

            //Dump RNN setting on console
            RNNConfig.DumpSetting();

            if (File.Exists(strFeatureConfigFile) == false)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The feature configuration file {0} isn't existed.", strFeatureConfigFile);
                UsageTrain();
                return;
            }
            //Create feature extractors and load word embedding data from file
            Featurizer featurizer = new Featurizer(strFeatureConfigFile, tagSet);
            featurizer.ShowFeatureSize();

            if (featurizer.IsRunTimeFeatureUsed() == true && iDir == 1)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: Run time feature is not available for bi-directional RNN model.");
                UsageTrain();
                return;
            }
            if (String.IsNullOrEmpty(strTrainFile) == true)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The training corpus isn't specified.");
                UsageTrain();
                return;
            }

            //LoadFeatureConfig training corpus and extract feature set
            DataSet dataSetTrain = new DataSet(tagSet.GetSize());
            LoadDataset(strTrainFile, featurizer, dataSetTrain);

            DataSet dataSetValidation = null;
            if (String.IsNullOrEmpty(strValidFile) == true)
            {
                Logger.WriteLine(Logger.Level.err, "FAILED: The validation corpus isn't specified.");
                return;
            }
            //LoadFeatureConfig validated corpus and extract feature set
            dataSetValidation = new DataSet(tagSet.GetSize());
            LoadDataset(strValidFile, featurizer, dataSetValidation);

            //Create RNN encoder and save necessary parameters
            RNNEncoder encoder = new RNNEncoder(RNNConfig);
            encoder.SetTrainingSet(dataSetTrain);
            encoder.SetValidationSet(dataSetValidation);

            if (iCRF == 1)
            {
                Logger.WriteLine(Logger.Level.info, "Initialize output tag bigram transition probability...");
                //Build tag bigram transition matrix
                dataSetTrain.BuildLabelBigramTransition();
                encoder.SetLabelBigramTransition(dataSetTrain.GetLabelBigramTransition());
            }

            //Start to train the model
            encoder.Train();

        }
    }
}
