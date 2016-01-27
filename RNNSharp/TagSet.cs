using System.Collections.Generic;
using System.IO;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class TagSet
    {
        public Dictionary<string, int> m_Tag2Index;
        public static string DefaultTag = "SentBE";

        public int GetSize()
        {
            return m_Tag2Index.Count;
        }

        public string GetTagName(int nIndex)
        {
            foreach (KeyValuePair<string, int> pair in m_Tag2Index)
            {
                if (pair.Value == nIndex)
                {
                    return pair.Key;
                }
            }

            return null;
        }

        public int GetIndex(string strTagName)
        {

            if (m_Tag2Index.ContainsKey(strTagName) == false)
            {
                return -1;
            }

            return m_Tag2Index[strTagName];
        }


        //Load tag name from given file
        //Format: each line has one tag name
        public TagSet(string strTagFileName)
        {
            m_Tag2Index = new Dictionary<string, int>();
            int idx = 0;
            m_Tag2Index.Add(DefaultTag, idx);
            idx++;

            string strLine = null;
            using (StreamReader fin = new StreamReader(strTagFileName))
            {
                while ((strLine = fin.ReadLine()) != null)
                {
                    strLine = strLine.Trim();
                    if (strLine.Length == 0)
                    {
                        continue;
                    }

                    m_Tag2Index.Add(strLine, idx);
                    idx++;
                }
            }
        }
    }
}
