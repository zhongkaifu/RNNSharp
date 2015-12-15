using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class neuron
    {
        public double cellOutput;		//actual value stored in neuron
        public double er;		//error value in neuron, used by learning algorithm
        public bool mask;

        public neuron CopyTo()
        {
            neuron neu = new neuron();
            neu.cellOutput = cellOutput;
            neu.er = er;
            neu.mask = mask;

            return neu;
        }
    }
}
