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

	private int nFeatures;        // The number of features including the class
	private int[] featureSize;    // Size of each features
	private List<List<Double[]>> logPosProbs;    // parameters of Naive Bayes

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
		int inst_count = data.length - 1;
		int pos_count = 0;
		double p_class;
		double v;
		for(int i = 0; i < nFeatures; i++) {
			for(int j = 0; j < featureSize[i]; j++) {

			}
		}
		//Prior probabilities
		//for each attribute get the number of values
		//compute P(attr = value) for each possible value
		//compute probability of a positive instance
		double denom = (inst_count + featureSize[featureSize.length - 1]);
		double num = pos_count + 1;
		p_class =  num / denom;
		System.out.println(p_class);
		//p_class = Math.log(p_class); //good
		List<Double[]> pos = new ArrayList<>(); //Hold the probabilities for a positive or negative classification
		Double[] y_pos = new Double[2];
		Double[] y_neg = new Double[2];
		y_neg[0] = Math.log((inst_count - pos_count + 1)/denom); // probability that the class is negative
		y_neg[1] = 0.0;
		y_pos[0] = 0.0;
		y_pos[1] = Math.log(p_class); // probability that the class is positive
		pos.add(0, y_neg);
		pos.add(1, y_pos);
		//logPosProbs.add(logPosProbs.size()- 1, pos);
		logPosProbs.add(pos);

		for (int i = 0; i < data[0].length - 1; i++) { //for each attribute
			List<Double[]> attribute_values = new ArrayList<>();
			//P(X = x | Y = y)
			//Getting #(x,y)
			//P(Y=0)
			List<Double[]> vals = new ArrayList<Double[]>(featureSize[i]);
			for (int j = 0; j < featureSize[i]; j++) {//for each value that the feature can take
				Double[] attr = new Double[2];
				int p_j_given_pos;
				int joint = 0;
				int njoint = 0;
				for (int k = 0; k < data.length; k++) { //for each instance
					if(data[k][i] == j) {
						if(data[k][data[k].length-1] == 1)
							joint++;
						else if (data[k][data[k].length-1] == 0)
							njoint++;
					}
				}
				double p_x_given_y = ((double)joint +1)/(p_class + featureSize[i]);
				double p_x_given_ny = ((double)njoint +1)/((1-p_class) + featureSize[i]);
				attr[0] = p_x_given_ny;
				attr[1] = p_x_given_y;
				vals.add(attr);
			}
			logPosProbs.add(vals);
		}
			/*
			for (int j = 0; j < featureSize[i]; j++) { //for each possible value of the current attribute
				int joint_pos = 0;
				int joint_neg = 0;
				Double[] attr_probs = new Double[2];
				//j starts as the highest possible value for the attribute and decreases to 0
				for (int k = 1; k < data.length; k++) { //for each instance
					if (data[k][i] == j && data[k][data[k].length - 1] == 1)
						joint_pos++;
					else if (data[k][i] == j && data[k][data[k].length - 1] == 0)
						joint_neg++;
				}
				//good
				double condit_prob = ((double)joint_pos + 1) / ((double)pos_count + featureSize[i]); //P(X = x|Y = 1)
				double not_condit = ((double)joint_neg + 1) / ((inst_count - pos_count) + featureSize[i]); //P(X = x|Y = 0)
				//attr_probs[0] = Math.log(not_condit);
				//attr_probs[1] = Math.log(condit_prob);
				//Place the probabilities in their appropriate places in the logPosProb structure
				//attribute_values.add(attr_probs);
			}
			logPosProbs.add(attribute_values);
			*/


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
		double pos = (logPosProbs.get(nFeatures - 1)).get(0)[1];
		Label[] yPred = new Label[nrows]; // predicted data
		double p_negative = logPosProbs.get(nFeatures-1).get(0)[0];
		double p_positive = logPosProbs.get(nFeatures-1).get(1)[1];
		for (int i = 0; i < instances.length; i++) { //for each instance
			int[] inst = instances[i];
			double p_pos = 0, p_neg = 0;
			for (int attr_ind = 0; attr_ind < nFeatures; attr_ind++) { //for each attribute in the instance
				List<Double[]> values = logPosProbs.get(attr_ind); //values for the attribute
				//probability that y is positive given the actual value of the attribute in inst
				int actual = inst[attr_ind];

				double p_actual_given_positive = values.get(actual)[1];
				p_pos +=  p_actual_given_positive;
				double p_actual_given_negative = values.get(actual)[0];
				p_neg += p_actual_given_negative;
				System.out.println("P(X = " + actual + "| Y=1): " + p_actual_given_positive);
			}
			p_neg += p_negative;
			p_pos += p_positive;
			System.out.println(p_pos + " " + p_neg);
			if (p_pos >= p_neg) yPred[i] = Label.Positive;
			else yPred[i] = Label.Negative;
			System.out.println(i + ". Actual: " + Label.values()[inst[nFeatures-1]] + " - Classified: " + yPred[i]);
		}
		return yPred;
	}
}