package co.echelon.echelonNN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.train.strategy.RequiredImprovementStrategy;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.neuroph.core.data.DataSet;

import com.aerospike.client.AerospikeClient;
import com.aerospike.client.AerospikeException;
import com.aerospike.client.Key;
import com.aerospike.client.Record;
import com.aerospike.client.ScanCallback;
import com.aerospike.client.policy.Policy;
import com.aerospike.client.policy.ScanPolicy;

public class App 
{
	static Policy universalReadPolicy;
	static DataSet trainingSet;
	static int inputSize;
	static int outputSize;
	static ArrayList<double[]> inputs;
	static ArrayList<double[]> targets;
	static String projectid;
	static int numEpochs;
	static int trainingAlgo;
	static String aeroAddr = "0.0.0.0";
	static int aeroPort = 3000;
	
	final static int TIMEOUT = 5000; //50 millisecond timeout
	final static String TEST_PROJECT = "testproject3"; 
	final static String AUTH_KEY = "jfvhaifhvrgtyvogrygtiavihaihifeadhshdvgey3e873y7w08a8ygf6"; 
	/***
	 * 
	 * @param args
	 * @throws IOException
	 * 
	 * args[0] projectid
	 * args[1] number of epochs
	 * args[2] training algorithm (1) Backprop (2) Rprop (3) Scaled Conjugate Gradient
	 * args[3] extra information LMT-->Number of Cycles
	 */
	
    public static void main( String[] args ) throws IOException
    {
    	projectid = args[0];
    	numEpochs = Integer.parseInt(args[1]);
    	trainingAlgo = Integer.parseInt(args[2]);
    	aeroAddr = args[3];
    	aeroPort = Integer.parseInt(args[4]);
    	
    	universalReadPolicy = new Policy();
    	universalReadPolicy.timeout = TIMEOUT;
    	
    	
    	System.out.println( "Connecting to cluster..." );
        AerospikeClient client = new AerospikeClient(aeroAddr,aeroPort);
        if(client.isConnected())
        {
        	System.out.println("client connected");
        	try
        	{
        		BasicNetwork nn = new BasicNetwork();
        		int[] nodeDist = getNetworkNodeDistribution(client, projectid);
        		for(int i = 0; i<nodeDist.length; i++)
        		{
        			nn.addLayer(new BasicLayer(new ActivationTANH(), true, nodeDist[i]));
        		}
        		
        		nn.getStructure().finalizeStructure();
        		
        		inputSize = nodeDist[0];
        		outputSize = nodeDist[nodeDist.length-1];
        		
        		inputs = new ArrayList<double[]>();
        		targets = new ArrayList<double[]>();
        		
        		pullData(client);
        		
        		double[][] real_inputs = new double[inputs.size()][inputs.get(0).length];
        		double[][] real_targets = new double[targets.size()][targets.get(0).length];
        		
        		inputs.toArray(real_inputs);
        		targets.toArray(real_targets);
        		
        		NeuralDataSet trainingSet = new BasicNeuralDataSet(real_inputs, real_targets);
        	
        		Train train;
        		
        		switch(trainingAlgo)
        		{
        			case 1:
        				train = new Backpropagation(nn, trainingSet);
        				break;
        			case 2:
        				train = new ResilientPropagation(nn, trainingSet);
        				break;
        			case 3:
        				train = new ScaledConjugateGradient(nn, trainingSet);
        			default:
        				train = new Backpropagation(nn, trainingSet);
        				break;
        		}
        		
        		train.addStrategy(new RequiredImprovementStrategy(5));
        		
        	
        		for(int epoch = 1; epoch<=numEpochs; epoch++)
        		{
        			train.iteration();
        		
        		}
        		
        		String result = "";

        	    for (int layer = 0; layer < nn.getLayerCount() - 1; layer++) {
        	        int bias = 0;
        	    

        	        for (int fromIdx = 0; fromIdx < nn.getLayerNeuronCount(layer); fromIdx++) {
        	            for (int toIdx = 0; toIdx < nn.getLayerNeuronCount(layer + 1); toIdx++) {
        	                String type1 = "", type2 = "";
        	                type1 = (layer)+"/"+fromIdx;
    	                    type2 = (layer) + "/"+toIdx;
        	                
        	                result+="{"+type1 + "," + type2
        	                        + "," + nn.getWeight(layer, fromIdx, toIdx)+","
        	                		+nn.getWeight(layer, nn.getLayerNeuronCount(layer), toIdx)
        	                        + "},";
        	            }
        	        }
        	    }
        	   result = result.substring(0, result.length()-1);
        	   String url = "http://echelon-nn.herokuapp.com/admin/projectops/pushWeights";
        	   
        	   ArrayList<BasicNameValuePair> params = new ArrayList<BasicNameValuePair>(4);
        	   params.add(new BasicNameValuePair("auth", AUTH_KEY));
        	   params.add(new BasicNameValuePair("projectid", projectid));
        	   params.add(new BasicNameValuePair("weights", result));
        	   params.add(new BasicNameValuePair("trainingError", ""+train.getError()));
        	   
        	   HttpPost httppost = new HttpPost(url);
        	   httppost.setEntity(new UrlEncodedFormEntity(params));
        	   HttpClient httpclient = new DefaultHttpClient();
        	   HttpResponse httpresponse = httpclient.execute(httppost);
        	   System.exit(0);
        	   
        	}
        	finally
        	{
        		client.close();
        	}
        }
        else
        {
        	System.out.println("Client failed to connect.");
        }     
    }
    
    public static void pullData(AerospikeClient client)
    {
    	ScanPolicy policy = new ScanPolicy();
		client.scanAll(policy, "dims", projectid, new ScanCallback()
		{
			public void scanCallback(Key key, Record record) throws AerospikeException
			{
				String[] str = record.getString("data").split(",");
				double[] _inputs = new double[inputSize];
				double[] _targets = new double[outputSize];
				
				for(int i = 0; i<inputSize; i++)
				{
					_inputs[i] = Integer.parseInt(str[i]);
				}
				for(int i = 0; i<outputSize; i++)
				{
					_targets[i] = Integer.parseInt(str[i+inputSize]);
				}
				
				inputs.add(_inputs);
				targets.add(_targets);
			}
		});	
    }
    
    public static int getNetworkLayerCount(AerospikeClient client, String projectid)
    {
    	Key key = new Key("pims","projectinfo" , projectid);
    	Record record = client.get(universalReadPolicy, key);
    	return Integer.parseInt(record.getString("num_layers"));
    }
    
    public static int[] getNetworkNodeDistribution(AerospikeClient client, String projectid)
    {
    	Key key = new Key("pims","projectinfo" , projectid);
    	Record record = client.get(universalReadPolicy, key);
    	String[] str = record.getString("neurons").split(",");
    	int[] nodesPerLayer = new int[str.length];
    	
    	for(int i  = 0; i<str.length; i++)
    	{
    		nodesPerLayer[i] = Integer.parseInt(str[i]);
    	}
    	
    	return nodesPerLayer;
    }
}
