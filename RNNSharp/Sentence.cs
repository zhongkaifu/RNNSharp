using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class Sentence
    {
        private List<string[]> m_features;

        public List<string[]> GetFeatureSet()
        {
            return m_features;
        }

        public int GetTokenSize()
        {
            return m_features.Count;
        }

        public void DumpFeatures()
        {
            foreach (string[] features in m_features)
            {
                StringBuilder sb = new StringBuilder();
                foreach (string strFeature in features)
                {
                    sb.Append(strFeature);
                    sb.Append('\t');
                }

                Console.WriteLine(sb.ToString().Trim());
            }
        }

        public virtual void SetFeatures(List<string> tokenList)
        {
            m_features = new List<string[]>();

            //Add the begining term for current record
            string[] curfeature = new string[2];
            curfeature[0] = "<s>";
            curfeature[1] = "O";
            m_features.Add(curfeature);

            foreach (string s in tokenList)
            {
                string[] tokens = s.Split('\t');
                m_features.Add(tokens);
            }

            //Add the end term of current record
            curfeature = new string[2];
            curfeature[0] = "</s>";
            curfeature[1] = "O";
            m_features.Add(curfeature);
        }



    }
}
