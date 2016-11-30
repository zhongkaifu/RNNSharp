using AdvUtils;
using System;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class Sequence : ISequence
    {
        public State[] States { get;}

        public int DenseFeatureSize
        {
            get
            {
                if (0 == States.Length || States[0].DenseFeature == null)
                {
                    return 0;
                }
                else
                {
                    return States[0].DenseFeature.Length;
                }
            }
        }

        public int SparseFeatureSize
        {
            get
            {
                if (0 == States.Length) return 0;
                else return States[0].SparseFeature.Length;
            }
        }

        public bool SetLabel(Sentence sent, TagSet tagSet)
        {
            List<string[]> tokensList = sent.TokensList;
            if (tokensList.Count != States.Length)
            {
                Logger.WriteLine(Logger.Level.warn,String.Format("Error: Inconsistent token({0}) and state({1}) size. Tokens list: {2}",
                    tokensList.Count, States.Length, sent.ToString()));
                return false;
            }

            for (int i = 0; i < tokensList.Count; i++)
            {
                string strTagName = tokensList[i][tokensList[i].Length - 1];
                int tagId = tagSet.GetIndex(strTagName);
                if (tagId < 0)
                {
                    Logger.WriteLine(Logger.Level.warn, String.Format("Error: tag {0} is unknown. Tokens list: {1}", 
                        strTagName, sent.ToString()));
                    return false;
                }

                States[i].Label = tagId;
            }

            return true;
        }

        public Sequence(int numStates)
        {
            States = new State[numStates];
            for (int i = 0; i < numStates; i++)
            {
                States[i] = new State();
            }
        }

    }
}
