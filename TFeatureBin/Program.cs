using AdvUtils;
using RNNSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace TFeatureBin
{
    internal class Program
    {
        private static string strTemplateFile = "";
        private static string strInputFile = "";
        private static string strFeatureFile = "";
        private static string strMode = "";
        private static int minfreq = 1;
        private static TemplateFeaturizer templateFeaturizer;

        private static int ArgPos(string str, string[] args)
        {
            int a;
            for (a = 0; a < args.Length; a++)
            {
                if (str == args[a])
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

        private static void UsageTitle()
        {
            Console.WriteLine("Template Feature Builder written by Zhongkai Fu(fuzhongkai@gmail.com)");
        }

        private static void Usage()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe <parameters>");
            Console.WriteLine("The tool is to generate template feature from corpus and index them into file");
            Console.WriteLine("-mode <string> : support extract,index and build modes, default is build mode.");
            Console.WriteLine("  extract : extract features from corpus and save them as raw text feature list");
            Console.WriteLine("  index   : build indexed feature set from raw text feature list");
            Console.WriteLine("  build   : extract features from corpus and generate indexed feature set");
        }

        private static void UsageBuild()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe -mode build <parameters>");
            Console.WriteLine("This mode is to extract feature from corpus and generate indexed feature set");
            Console.WriteLine("-template <string> : feature template file");
            Console.WriteLine("-inputfile <string> : file used to generate features");
            Console.WriteLine("-ftrfile <string> : generated indexed feature file");
            Console.WriteLine("-minfreq <int> : min-frequency of feature");
            Console.WriteLine();
            Console.WriteLine(
                "Example: TFeatureBin.exe -mode build -template template.txt -inputfile train.txt -ftrfile tfeatures -minfreq 3");
        }

        private static void UsageExtract()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe -mode extract <parameters>");
            Console.WriteLine("This mode is to extract features from corpus and save them as text feature list");
            Console.WriteLine("-template <string> : feature template file");
            Console.WriteLine("-inputfile <string> : file used to generate features");
            Console.WriteLine("-ftrfile <string> : generated feature list file in raw text format");
            Console.WriteLine("-minfreq <int> : min-frequency of feature");
            Console.WriteLine();
            Console.WriteLine(
                "Example: TFeatureBin.exe -mode extract -template template.txt -inputfile train.txt -ftrfile features_raw.txt -minfreq 3");
        }

        private static void UsageIndex()
        {
            UsageTitle();
            Console.WriteLine("TFeatureBin.exe -mode index <parameters>");
            Console.WriteLine("This mode is to build indexed feature set from raw text feature list");
            Console.WriteLine("-template <string> : feature template file");
            Console.WriteLine("-inputfile <string> : feature list in raw text format");
            Console.WriteLine("-ftrfile <string> : indexed feature set");
            Console.WriteLine();
            Console.WriteLine(
                "Example: TFeatureBin.exe -mode index -template template.txt -inputfile features.txt -ftrfile tfeatures");
        }

        private static void ExtractMode()
        {
            if (File.Exists(strInputFile) == false ||
                File.Exists(strTemplateFile) == false)
            {
                UsageExtract();
                return;
            }

            //Extract feature set from given corpus
            var feature2freq = ExtractFeatureSetFromFile();

            //Save feature set into raw text file
            var sw = new StreamWriter(strFeatureFile, false, Encoding.UTF8);
            foreach (var pair in feature2freq)
            {
                sw.WriteLine("{0}\t{1}", pair.Key, pair.Value);
            }
            sw.Close();
        }

        private static void IndexMode()
        {
            if (File.Exists(strInputFile) == false ||
                File.Exists(strTemplateFile) == false)
            {
                UsageIndex();
                return;
            }

            //Load feature set from given file
            var features = new List<string>();
            var sr = new StreamReader(strInputFile);
            string strLine;

            while ((strLine = sr.ReadLine()) != null)
            {
                var items = strLine.Split('\t');
                features.Add(items[0]);
            }
            sr.Close();

            //Build indexed feature set
            templateFeaturizer = new TemplateFeaturizer();
            templateFeaturizer.LoadTemplateFromFile(strTemplateFile);
            templateFeaturizer.BuildIndexedFeatureIntoFile(strFeatureFile, features);
        }

        private static void BuildMode()
        {
            if (File.Exists(strInputFile) == false ||
                File.Exists(strTemplateFile) == false)
            {
                UsageBuild();
                return;
            }

            //Extract feature set from given corpus
            var feature2freq = ExtractFeatureSetFromFile();
            var features = feature2freq.Select(pair => pair.Key).ToList();

            //Build indexed feature set
            templateFeaturizer.BuildIndexedFeatureIntoFile(strFeatureFile, features);
        }

        private static IDictionary<string, int> ExtractFeatureSetFromFile()
        {
            //Load templates from given file
            Logger.WriteLine("Loading feature template from {0}...", strTemplateFile);
            templateFeaturizer = new TemplateFeaturizer();
            templateFeaturizer.LoadTemplateFromFile(strTemplateFile);

            Logger.WriteLine("Generate feature set...");
            var feature2freq = new BigDictionary<string, int>();

            var tokenList = new List<string[]>();

            using (var srCorpus = new StreamReader(strInputFile, Encoding.UTF8))
            {
                string strLine;
                Sentence sentence;
                while ((strLine = srCorpus.ReadLine()) != null)
                {
                    strLine = strLine.Trim();
                    if (strLine.Length == 0)
                    {
                        //The end of current record
                        sentence = new Sentence(tokenList);
                        for (var i = 0; i < sentence.TokensList.Count; i++)
                        {
                            //Get feature of i-th token
                            var featureList = templateFeaturizer.GenerateFeature(sentence.TokensList, i);
                            foreach (var strFeature in featureList)
                            {
                                if (feature2freq.ContainsKey(strFeature) == false)
                                {
                                    feature2freq.Add(strFeature, 0);
                                }
                                feature2freq[strFeature]++;
                            }
                        }

                        tokenList.Clear();
                    }
                    else
                    {
                        tokenList.Add(strLine.Split('\t'));
                    }
                }

                //The end of current record
                sentence = new Sentence(tokenList);
                for (var i = 0; i < sentence.TokensList.Count; i++)
                {
                    //Get feature of i-th token
                    var featureList = templateFeaturizer.GenerateFeature(sentence.TokensList, i);
                    foreach (var strFeature in featureList)
                    {
                        if (feature2freq.ContainsKey(strFeature) == false)
                        {
                            feature2freq.Add(strFeature, 0);
                        }
                        feature2freq[strFeature]++;
                    }
                }
            }

            //Only save the feature whose frequency is not less than minfreq
            Logger.WriteLine("Filter out features whose frequency is less than {0}", minfreq);
            var features = new SortedDictionary<string, int>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, int> pair in feature2freq)
            {
                if (pair.Value >= minfreq)
                {
                    features.Add(pair.Key, pair.Value);
                }
            }

            return features;
        }

        private static void Main(string[] args)
        {
            int i;
            if ((i = ArgPos("-template", args)) >= 0) strTemplateFile = args[i + 1];
            if ((i = ArgPos("-inputfile", args)) >= 0) strInputFile = args[i + 1];
            if ((i = ArgPos("-ftrfile", args)) >= 0) strFeatureFile = args[i + 1];
            if ((i = ArgPos("-minfreq", args)) >= 0) minfreq = int.Parse(args[i + 1]);
            if ((i = ArgPos("-mode", args)) >= 0) strMode = args[i + 1];

            switch (strMode)
            {
                case "build":
                    BuildMode();
                    break;

                case "extract":
                    ExtractMode();
                    break;

                case "index":
                    IndexMode();
                    break;

                default:
                    Usage();
                    break;
            }
        }
    }
}