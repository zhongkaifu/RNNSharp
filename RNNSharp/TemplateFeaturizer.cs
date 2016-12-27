using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    //Template feature processor
    public class TemplateFeaturizer
    {
        private DoubleArrayTrieSearch daSearch;

        public int m_maxFeatureId;
        private List<string> m_Templates;

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
            var featureList = GenerateFeature(record, startX);

            //Check if the feature string has already built into feature set,
            //If yes, save the feature id, otherwise, ignore the feature

            return
                featureList.Select(strFeature => daSearch.SearchByPerfectMatch(strFeature))
                    .Where(id => id >= 0)
                    .ToList();
        }

        //Generate feature in string by given record, start position and saved templates
        public List<string> GenerateFeature(List<string[]> record, int startX)
        {
            return m_Templates.Select(strTemplate => GenerateFeature(strTemplate, record, startX)).ToList();
        }

        //U22:%x[-4,0]/%x[-3,0]/%x[-2,0]
        private string GenerateFeature(string strTemplate, List<string[]> record, int startX)
        {
            var sb = new StringBuilder();

            var keyvalue = strTemplate.Split(':');
            var tId = keyvalue[0];
            sb.Append(tId);
            sb.Append(":");

            var items = keyvalue[1].Split('/');
            foreach (var item in items)
            {
                var bpos = item.LastIndexOf('[');
                var enpos = item.LastIndexOf(']');
                var strPos = item.Substring(bpos + 1, enpos - bpos - 1);
                var xy = strPos.Split(',');
                var x = int.Parse(xy[0]) + startX;
                var y = int.Parse(xy[1]);

                if (x >= 0 && x < record.Count &&
                    y >= 0 && y < record[x].Length)
                {
                    sb.Append(record[x][y]);
                }
                else
                {
                    if (x < 0)
                    {
                        sb.Append("B_" + x + "_" + xy[1]);
                    }
                    else
                    {
                        sb.Append("B_" + (x - record.Count + 1) + "_" + xy[1]);
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

            var sr = new StreamReader(strFileName);
            string strLine;
            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.StartsWith("#"))
                {
                    //Ignore comment line
                    continue;
                }

                //Only load U templates
                if (strLine.StartsWith("U"))
                {
                    m_Templates.Add(strLine);
                }
                else if (strLine.StartsWith("MaxTemplateFeatureId:"))
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
            var feature2Id = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var maxId = 0;
            foreach (var strFeature in features)
            {
                if (strFeature.StartsWith("U") == false)
                {
                    Logger.WriteLine("Invalidated feature: {0}", strFeature);
                    continue;
                }

                feature2Id.Add(strFeature, maxId);
                maxId++;
            }

            var da = new DoubleArrayTrieBuilder(4);
            da.build(feature2Id);
            da.save(strFileName + ".dart");

            var swTemplate = new StreamWriter(strFileName + ".template");
            swTemplate.WriteLine("MaxTemplateFeatureId:{0}", maxId);
            foreach (var strTemplate in m_Templates)
            {
                swTemplate.WriteLine(strTemplate);
            }

            swTemplate.Close();
        }
    }
}