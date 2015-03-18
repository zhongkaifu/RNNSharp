using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class TagSet
    {
        public Dictionary<string, int> m_Tag2Index = new Dictionary<string, int>();

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


        //Load the tag id and its name mapping from given file
        //Format: tagid /t tag name
        public TagSet(string strTagFileName)
        {
            StreamReader fin = new StreamReader(strTagFileName);

            int idx;
            string strTagName;
            string strLine = null;
            while ((strLine = fin.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.Length == 0)
                {
                    continue;
                }

                string[] items = strLine.Split('\t');
                idx = int.Parse(items[0]);
                strTagName = items[1];

                m_Tag2Index.Add(strTagName, idx);
            }
            fin.Close();
        }
    }
}
