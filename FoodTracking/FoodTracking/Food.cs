using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Newtonsoft.Json;

namespace FoodTracking
{
    public class Food
    {
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("fill_rate")]
        public double FillRate { get; set; }

        [JsonProperty("message")]
        public string Message { get; set; }
    }
}

