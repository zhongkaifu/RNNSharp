
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public struct neuron
    {
        public double cellOutput;		//actual value stored in neuron
        public double er;		//error value in neuron, used by learning algorithm
        public bool mask;
    }
}
