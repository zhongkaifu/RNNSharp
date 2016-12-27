using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class Sequence : ISequence
    {
        public Sequence(int numStates)
        {
            States = new State[numStates];
            for (var i = 0; i < numStates; i++)
            {
                States[i] = new State();
            }
        }

        public State[] States { get; }

        public int DenseFeatureSize
        {
            get
            {
                if (0 == States.Length || States[0].DenseFeature == null)
                {
                    return 0;
                }
                return States[0].DenseFeature.Length;
            }
        }

        public int SparseFeatureSize => 0 == States.Length ? 0 : States[0].SparseFeature.Length;

        public bool SetLabel(Sentence sent, TagSet tagSet)
        {
            var tokensList = sent.TokensList;
            if (tokensList.Count != States.Length)
            {
                Logger.WriteLine(Logger.Level.warn,
                    $"Error: Inconsistent token({tokensList.Count}) and state({States.Length}) size. Tokens list: {sent}");
                return false;
            }

            for (var i = 0; i < tokensList.Count; i++)
            {
                var strTagName = tokensList[i][tokensList[i].Length - 1];
                var tagId = tagSet.GetIndex(strTagName);
                if (tagId < 0)
                {
                    Logger.WriteLine(Logger.Level.warn, $"Error: tag {strTagName} is unknown. Tokens list: {sent}");
                    return false;
                }

                States[i].Label = tagId;
            }

            return true;
        }
    }
}