using AdvUtils;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class TagSet
    {
        public static string DefaultTag = "SentBE";
        public Dictionary<string, int> m_Tag2Index;

        //Load tag name from given file
        //Format: each line has one tag name
        public TagSet(string strTagFileName)
        {
            m_Tag2Index = new Dictionary<string, int>();
            var idx = 0;
            m_Tag2Index.Add(DefaultTag, idx);
            idx++;

            using (var fin = new StreamReader(strTagFileName))
            {
                string strLine;
                while ((strLine = fin.ReadLine()) != null)
                {
                    strLine = strLine.Trim();
                    if (strLine.Length == 0)
                    {
                        continue;
                    }

                    if (m_Tag2Index.ContainsKey(strLine))
                    {
                        Logger.WriteLine($"Character '{strLine}' (index = '{idx}') is duplicated in tag set.");
                    }
                    else
                    {
                        m_Tag2Index.Add(strLine, idx);
                        idx++;
                    }
                }
            }
        }

        public int GetSize()
        {
            return m_Tag2Index.Count;
        }

        public string GetTagName(int nIndex)
        {
            return (from pair in m_Tag2Index where pair.Value == nIndex select pair.Key).FirstOrDefault();
        }

        public int GetIndex(string strTagName)
        {
            if (m_Tag2Index.ContainsKey(strTagName) == false)
            {
                return -1;
            }

            return m_Tag2Index[strTagName];
        }
    }
}