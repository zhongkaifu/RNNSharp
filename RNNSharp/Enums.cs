using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public enum TFEATURE_WEIGHT_TYPE_ENUM
    {
        BINARY,
        FREQUENCY
    }

    public enum LayerType
    {
        Softmax,
        NCESoftmax,
        BPTT,
        LSTM,
        DropOut,
        None
    }

    public enum PRETRAIN_TYPE
    {
        AutoEncoder,
        Embedding
    }

    public enum MODELDIRECTION
    {
        Forward = 0,
        BiDirectional
    }

    public enum MODELTYPE
    {
        SeqLabel = 0,
        Seq2Seq
    }

    public enum LAYERTYPE
    {
        BPTT = 0,
        LSTM
    }

    public enum RunningMode
    {
        Training = 0,
        Validate = 1,
        Test = 2
    }
}
