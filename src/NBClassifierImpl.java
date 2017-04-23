/***************************************************************************************
 CS540 - Section 2
 Homework Assignment 5: Naive Bayes

 NBClassifierImpl.java
 This is the main class that implements functions for Naive Bayes Algorithm!
 ---------
 *Free to modify anything in this file, except the class name
 You are required:
 - To keep the class name as NBClassifierImpl for testing
 - Not to import any external libraries
 - Not to include any packages
 *Notice: To use this file, you should implement 2 methods below.

 @author: TA
 @date: April 2017
 *****************************************************************************************/

import java.util.ArrayList;
import java.util.List;


public class NBClassifierImpl implements NBClassifier {

	private int nFeatures; 		// The number of features including the class 
	private int[] featureSize;	// Size of each features
	private List<List<Double[]>> logPosProbs;	// parameters of Naive Bayes

	/**
	 * Constructs a new classifier without any trained knowledge.
	 */
	public NBClassifierImpl() {

	}

	/**
	 * Construct a new classifier 
	 *
	 * @param features int[] sizes of all attributes
	 */
	public NBClassifierImpl(int[] features) {
		this.nFeatures = features.length;

		// initialize feature size
		this.featureSize = features.clone();

		this.logPosProbs = new ArrayList<List<Double[]>>(this.nFeatures);
	}


	/**
	 * Read training data and learn parameters
	 *
	 * @param data training data
	 */
	@Override
	public void fit(int[][] data) {
		int inst_count = data.length - 2;
		int class_count = 0;
		double p_class;
		double v;
		//Prior probabilities
		//for each attribute get the number of values
		//compute P(attr = value) for each possible value
		//compute probability of a positive instance
		for(int i = 1; i < data.length - 1; i++) {
			if (data[i][data[i].length - 1] == 1)
				class_count++;
		}
		p_class = (class_count + 1)/(inst_count + featureSize[featureSize.length-1]);
		Math.log(p_class);
		List<Double[]> pos = new ArrayList<>();
		Double [] y_pos = new Double[1];
		Double [] y_neg = new Double[1];
		y_neg[0] = 1-p_class;
		y_pos[0] = p_class;
		pos.add(0, y_neg);
		pos.add(1, y_pos);
		//probabilities given positive class
		for(int i = 0; i < data[0].length - 1; i++) { //for each attribute
			List<Double[]> attribute_values = new ArrayList<>();
			//P(X = x | Y = y)
			//Getting #(x,y)
			Double[] attr_probs = new Double[2];
			int joint_count = 0;
			for(int j = featureSize[i]; j >= 0; j--) { //for each possible value of the current attribute
				for(int k = 1; k < data.length-1; k++) { //for each instance
					if(data[k][i] == j && data[k][data[k].length-1] == 1)
						joint_count++;
				}
				double condit_prob =  (joint_count + 1)/(class_count + featureSize[i]); //P(X = x|Y = 1)
				double not_condit = (joint_count + 1)/((inst_count - class_count) + featureSize[i]); //P(X = x|Y = 0)
				attr_probs[1] = Math.log(condit_prob);
				attr_probs[0] = Math.log(not_condit);
				//Place the probabilities in their appropriate places in the logPosProb structure
				attribute_values.add(j, attr_probs);
			}
			logPosProbs.add(i, attribute_values);
		}
	}

	/**
	 * Classify new dataset
	 *
	 * @param instances int[][] test data
	 * @return Label[] classified labels
	 */
	@Override
	public Label[] classify(int[][] instances) {
		int nrows = instances.length;
		double pos = ((logPosProbs.get(nFeatures-1)).get(0))[1];
		Label[] yPred = new Label[nrows]; // predicted data
		//for each instance
		for(int i = 0; i < instances.length; i++) {
			//for each attribute of the instance
			double most_probable = 0;
			for(int j = 0; j < instances[i].length; j++) {
				for(int k = 0; k < featureSize[j]; k++) {
					//P(Y=v)Product(P(Xj = xj|Y = v)
					double positive = pos*(logPosProbs.get(j)).get(k)[0];
					double negative = (1-pos)*(logPosProbs.get(j)).get(k)[1];
					if(positive > most_probable)
						most_probable = positive;
					if(negative > most_probable)
						most_probable = negative;
				}

			}
		}
		//TODO
		return yPred;
	}
}