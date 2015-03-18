using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;
using RNNSharp;

namespace TFeatureBin
{
    class Program
    {
        static string strTemplateFile = "";
        static string strInputFile = "";
        static string strFeatureFile = "";
        static string strMode = "";
        static int minfreq = 1;
        static TemplateFeaturizer templateFeaturizer;

        static int ArgPos(string str, string[] args)
        {
            int a;
            for (a = 0; a < args.Length; a++)
            {
                if (str == args[a])
                {
                    if (a == args.Length - 1)
                    {
                        Console.WriteLine("Argument missing for {0}", str);
                        return -1;
                    }
                    return a;
                }
            }
            return -1;
        }

        static void UsageTitle()
        {
            Console.WriteLine("Template Feature Builder written by Zhongkai Fu(fuzhongkai@gmail.com)");
        }
        static void Usage()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe <parameters>");
            Console.WriteLine("The tool is to generate template feature from corpus and index them into file");
            Console.WriteLine("-mode <string> : support extract,index and build modes, default is build mode.");
            Console.WriteLine("  extract : extract features from corpus and save them as raw text feature list");
            Console.WriteLine("  index   : build indexed feature set from raw text feature list");
            Console.WriteLine("  build   : extract features from corpus and generate indexed feature set");
        }

        static void UsageBuild()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe -mode build <parameters>");
            Console.WriteLine("This mode is to extract feature from corpus and generate indexed feature set");
            Console.WriteLine("-template <string> : feature template file");
            Console.WriteLine("-inputfile <string> : file used to generate features");
            Console.WriteLine("-ftrfile <string> : generated indexed feature file");
            Console.WriteLine("-minfreq <int> : min-frequency of feature");
            Console.WriteLine();
            Console.WriteLine("Example: TFeatureBin.exe -mode build -template template.txt -inputfile train.txt -ftrfile tfeatures -minfreq 3");
        }

        static void UsageExtract()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe -mode extract <parameters>");
            Console.WriteLine("This mode is to extract features from corpus and save them as text feature list");
            Console.WriteLine("-template <string> : feature template file");
            Console.WriteLine("-inputfile <string> : file used to generate features");
            Console.WriteLine("-ftrfile <string> : generated feature list file in raw text format");
            Console.WriteLine("-minfreq <int> : min-frequency of feature");
            Console.WriteLine();
            Console.WriteLine("Example: TFeatureBin.exe -mode extract -template template.txt -inputfile train.txt -ftrfile features_raw.txt -minfreq 3");
        }

        static void UsageIndex()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe -mode index <parameters>");
            Console.WriteLine("This mode is to build indexed feature set from raw text feature list");
            Console.WriteLine("-template <string> : feature template file");
            Console.WriteLine("-inputfile <string> : feature list in raw text format");
            Console.WriteLine("-ftrfile <string> : indexed feature set");
            Console.WriteLine();
            Console.WriteLine("Example: TFeatureBin.exe -mode index -template template.txt -inputfile features.txt -ftrfile tfeatures");
        }

        static void ExtractMode()
        {
            if (File.Exists(strInputFile) == false ||
                File.Exists(strTemplateFile) == false)
            {
                UsageExtract();
                return;
            }

            //Extract feature set from given corpus
            IDictionary<string, int> feature2freq = ExtractFeatureSetFromFile();

            //Save feature set into raw text file
            StreamWriter sw = new StreamWriter(strFeatureFile, false, Encoding.UTF8);
            foreach (KeyValuePair<string, int> pair in feature2freq)
            {
                sw.WriteLine("{0}\t{1}", pair.Key, pair.Value);
            }
            sw.Close();
        }

        static void IndexMode()
        {
            if (File.Exists(strInputFile) == false ||
                File.Exists(strTemplateFile) == false)
            {
                UsageIndex();
                return;
            }

            //Load feature set from given file
            List<string> features = new List<string>();
            StreamReader sr = new StreamReader(strInputFile);
            string strLine = null;

            while ((strLine = sr.ReadLine()) != null)
            {
                string[] items = strLine.Split('\t');
                features.Add(items[0]);
            }
            sr.Close();

            //Build indexed feature set
            templateFeaturizer = new TemplateFeaturizer();
            templateFeaturizer.LoadTemplateFromFile(strTemplateFile);
            templateFeaturizer.BuildIndexedFeatureIntoFile(strFeatureFile, features);
        }

        static void BuildMode()
        {
            if (File.Exists(strInputFile) == false ||
                File.Exists(strTemplateFile) == false)
            {
                UsageBuild();
                return;
            }

            //Extract feature set from given corpus
            IDictionary<string, int> feature2freq = ExtractFeatureSetFromFile();
            List<string> features = new List<string>();
            foreach (KeyValuePair<string, int> pair in feature2freq)
            {
                features.Add(pair.Key);
            }

            //Build indexed feature set
            templateFeaturizer.BuildIndexedFeatureIntoFile(strFeatureFile, features);
        }

        static IDictionary<string, int> ExtractFeatureSetFromFile()
        {
            //Load templates from given file
            Console.WriteLine("Loading feature template from {0}...", strTemplateFile);
            templateFeaturizer = new TemplateFeaturizer();
            templateFeaturizer.LoadTemplateFromFile(strTemplateFile);

            Console.WriteLine("Generate feature set...");
            BigDictionary<string, int> feature2freq = new BigDictionary<string, int>();
            List<string[]> record = new List<string[]>();
            StreamReader srCorpus = new StreamReader(strInputFile, Encoding.UTF8);
            string strLine = null;
            while ((strLine = srCorpus.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.Length == 0)
                {
                    //The end of current record
                    for (int i = 0; i < record.Count; i++)
                    {
                        //Get feature of current token
                        List<string> featureList = templateFeaturizer.GenerateFeature(record, i);
                        foreach (string strFeature in featureList)
                        {
                            if (feature2freq.ContainsKey(strFeature) == false)
                            {
                                feature2freq.Add(strFeature, 0);
                            }
                            feature2freq[strFeature]++;
                        }
                    }

                    record.Clear();
                }
                else
                {
                    string[] items = strLine.Split('\t');
                    record.Add(items);
                }

            }

            //The end of current record
            for (int i = 0; i < record.Count; i++)
            {
                //Get feature of current token
                List<string> featureList = templateFeaturizer.GenerateFeature(record, i);
                foreach (string strFeature in featureList)
                {
                    if (feature2freq.ContainsKey(strFeature) == false)
                    {
                        feature2freq.Add(strFeature, 0);
                    }
                    feature2freq[strFeature]++;
                }
            }

            srCorpus.Close();

            //Only save the feature whose frequency is not less than minfreq
            Console.WriteLine("Filter out features whose frequency is less than {0}", minfreq);
            SortedDictionary<string, int> features = new SortedDictionary<string, int>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, int> pair in feature2freq)
            {
                if (pair.Value >= minfreq)
                {
                    features.Add(pair.Key, pair.Value);
                }
            }

            return features;
        }

        static void Main(string[] args)
        {
            int i;
            if ((i = ArgPos("-template", args)) >= 0) strTemplateFile = args[i + 1];
            if ((i = ArgPos("-inputfile", args)) >= 0) strInputFile = args[i + 1];
            if ((i = ArgPos("-ftrfile", args)) >= 0) strFeatureFile = args[i + 1];
            if ((i = ArgPos("-minfreq", args)) >= 0) minfreq = int.Parse(args[i + 1]);
            if ((i = ArgPos("-mode", args)) >= 0) strMode = args[i + 1];

            if (strMode == "build")
            {
                BuildMode();
            }
            else if (strMode == "extract")
            {
                ExtractMode();
            }
            else if (strMode == "index")
            {
                IndexMode();
            }
            else
            {
                Usage();
            }
        }
    }
}
