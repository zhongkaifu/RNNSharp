using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    //Template feature processor
    public class TemplateFeaturizer
    {
        List<string> m_Templates;
        DoubleArrayTrieSearch daSearch;
        
        public int m_maxFeatureId = 0;

        public TemplateFeaturizer()
        {

        }

        public TemplateFeaturizer(string strFileName)
        {
            LoadTemplateFromFile(strFileName + ".template");
            LoadFeatureFromFile(strFileName + ".dart");
        }

        public int GetFeatureSize()
        {
            return m_maxFeatureId;
        }

        //Extract feature id list from given record and start position
        public List<int> GetFeatureIds(List<string[]> record, int startX)
        {
            //Get the feature string
            List<string> featureList = GenerateFeature(record, startX);
            List<int> featureIdList = new List<int>();

            //Check if the feature string has already built into feature set, 
            //If yes, save the feature id, otherwise, ignore the feature
            foreach (string strFeature in featureList)
            {
                int id = daSearch.SearchByPerfectMatch(strFeature);
                if (id >= 0)
                {
                    featureIdList.Add(id);
                }
            }

            return featureIdList;
        }

        //Generate feature in string by given record, start position and saved templates
        public List<string> GenerateFeature(List<string[]> record, int startX)
        {
            List<string> featureList = new List<string>();
            foreach (string strTemplate in m_Templates)
            {
                //Generate feature by template
                string strFeature = GenerateFeature(strTemplate, record, startX);
                featureList.Add(strFeature);
            }

            return featureList;
        }

        //U22:%x[-4,0]/%x[-3,0]/%x[-2,0]
        private string GenerateFeature(string strTemplate, List<string[]> record, int startX)
        {
            StringBuilder sb = new StringBuilder();

            string[] keyvalue = strTemplate.Split(':');
            string tId = keyvalue[0];
            sb.Append(tId);
            sb.Append(":");

            string[] items = keyvalue[1].Split('/');
            foreach (string item in items)
            {
                int bpos = item.LastIndexOf('[');
                int enpos = item.LastIndexOf(']');
                string strPos = item.Substring(bpos + 1, enpos - bpos - 1);
                string[] xy = strPos.Split(',');
                int x = int.Parse(xy[0]) + startX;
                int y = int.Parse(xy[1]);

                if (x >= 0 && x < record.Count &&
                    y >= 0 && y < record[x].Length)
                {
                    sb.Append(record[x][y]);
                }
                else
                {
                    if (x < 0)
                    {
                        sb.Append("B_" + x.ToString() + "_" + xy[1]);
                    }
                    else
                    {
                        sb.Append("B_" + (x - record.Count + 1).ToString() + "_" + xy[1]);
                    }
                }
                sb.Append("/");
            }

            sb.Remove(sb.Length - 1, 1);

            return sb.ToString();
        }

        //Load template from given file
        public void LoadTemplateFromFile(string strFileName)
        {
            m_Templates = new List<string>();

            StreamReader sr = new StreamReader(strFileName);
            string strLine = null;
            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.StartsWith("#") == true)
                {
                    //Ignore comment line
                    continue;
                }

                //Only load U templates
                if (strLine.StartsWith("U") == true)
                {
                    m_Templates.Add(strLine);
                }
                else if (strLine.StartsWith("MaxTemplateFeatureId:") == true)
                {
                    strLine = strLine.Replace("MaxTemplateFeatureId:", "");
                    m_maxFeatureId = int.Parse(strLine);
                }
            }
            sr.Close();
        }

        private void LoadFeatureFromFile(string strFileName)
        {
            daSearch = new DoubleArrayTrieSearch();
            daSearch.Load(strFileName);
        }

        public void BuildIndexedFeatureIntoFile(string strFileName, List<string> features)
        {
            //Assign id for each feature
            SortedDictionary<string, int> feature2Id = new SortedDictionary<string, int>(StringComparer.Ordinal);
            int maxId = 0;
            foreach (string strFeature in features)
            {
                if (strFeature.StartsWith("U") == false)
                {
                    Logger.WriteLine("Invalidated feature: {0}", strFeature);
                    continue;
                }

                feature2Id.Add(strFeature, maxId);
                maxId++;
            }

            DoubleArrayTrieBuilder da = new DoubleArrayTrieBuilder(4);
            da.build(feature2Id);
            da.save(strFileName + ".dart");


            StreamWriter swTemplate = new StreamWriter(strFileName + ".template");
            swTemplate.WriteLine("MaxTemplateFeatureId:{0}", maxId);
            foreach (string strTemplate in m_Templates)
            {
                swTemplate.WriteLine(strTemplate);
            }

            swTemplate.Close();
        }
    }
}
