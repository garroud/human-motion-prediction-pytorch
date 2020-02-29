import sys
sys.path.append('src/')

from datasets.dataset import Dataset
import human_motion.data_utils as data_utils
import torch
import numpy as np

## include all the functionality for Human motion dataset
class HumanDataset(Dataset):
    def __init__(self,
        action,
        seq_length_in,
        seq_length_out,
        encoder_input_size,
        decoder_input_size,
        decoder_output_size,
        num_joint,
        data_dir,
        one_hot,
        use_GNN,
        device,
        dtype=torch.float32,
    ):
        super(HumanDataset, self).__init__()
        self.actions = self.define_actions(action)
        self.train_set, self.test_set, self.data_mean, self.data_std, self.dim_to_ignore, self.dim_to_use = \
            self.read_all_data(self.actions, seq_length_in, seq_length_out,
            data_dir, one_hot, use_GNN)
        self.seq_length_in = seq_length_in
        self.seq_length_out = seq_length_out
        self.one_hot = one_hot
        self.use_GNN = use_GNN
        self.encoder_input_size = encoder_input_size
        self.decoder_input_size = decoder_input_size
        self.decoder_output_size = decoder_output_size
        if self.one_hot:
            self.encoder_input_size += len(self.actions)
            self.decoder_input_size += len(self.actions)
        self.num_joint = num_joint
        self.dtype = dtype
        self.device = device

    def get_batch_train(self, batch_size):
        return self.get_batch(self.train_set, batch_size=batch_size, validation=False)

    def get_batch_validation(self):
        # batch size is useless here
        return self.get_batch(self.test_set, batch_size=1,validation=True)

    def get_batch_test(self):
        print("Useless call for human motion dataset")
        pass

    # get batch functions according to the seq2seq model, reformating the data
    def get_batch(self, data, batch_size, validation=False):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
          encoder_input : batch_size * num_joint * (seq_len * 3)
          decoder_input : batch_size * seq_len * num_joint * 3
          decoder_output : batch_size * seq_len * input_size
        """
        all_keys    = list(data.keys())
        if validation:
            chosen_keys = list(range(0,30,2))
        # Select entries at random
        else:
            chosen_keys = np.random.choice( len(all_keys), batch_size)
        # chosen_keys = [0] * self.batch_size
        # How many frames in total do we need?
        total_frames = self.seq_length_in + self.seq_length_out
        batch_size = len(chosen_keys)

        encoder_inputs  = np.zeros((batch_size, self.num_joint, self.encoder_input_size), dtype=float)
        decoder_inputs  = np.zeros((batch_size, self.seq_length_out, self.num_joint, self.decoder_input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, self.seq_length_out, self.decoder_output_size), dtype=float)

        for i in range(batch_size):

          the_key = all_keys[ chosen_keys[i] ]

          # Get the number of frames
          n, _ = data[ the_key ].shape

          # Sample somewherein the middle
          if validation:
              idx = 17
          else:
              idx = np.random.randint( 16, n-total_frames )
          # idx = 17

          # Select the data around the sampled points
          data_sel = data[ the_key ][idx:idx+total_frames ,:]
          data_to_transform = data_sel
          # get the class info
          if self.one_hot:
              class_encoding = data_sel[0,-len(self.actions):]
              data_to_transform = data_sel[:,:-len(self.actions)]
              encoder_inputs[i,:,-len(self.actions):] = class_encoding
              decoder_inputs[i,:,:,-len(self.actions):] = class_encoding
          # do the transfrom
          encoder_input = np.reshape(
            data_to_transform[0:self.seq_length_in,:], [-1, self.num_joint, 3])
          encoder_input =  np.reshape(np.transpose(encoder_input, [1,0,2]),[self.num_joint,-1])
          decoder_input = np.reshape(
            data_to_transform[self.seq_length_in-1:total_frames-1, :],
            [-1, self.num_joint, 3]
          )
          decoder_output = data_to_transform[self.seq_length_in:, :]

          # Add the data
          if self.one_hot:
              encoder_inputs[i,:,0:-len(self.actions)]  = encoder_input
              decoder_inputs[i,:,:,0:-len(self.actions)]  = decoder_input
          else:
              encoder_inputs[i] = encoder_input
              decoder_inputs[i] = decoder_input
          decoder_outputs[i] = decoder_output

          # alter data to expected form
        encoder_inputs = torch.tensor(encoder_inputs,dtype=self.dtype).to(self.device)
        decoder_inputs = torch.tensor(decoder_inputs,dtype=self.dtype).permute(1,0,2,3).to(self.device)
        decoder_outputs = torch.tensor(decoder_outputs,dtype=self.dtype).permute(1,0,2).to(self.device)

        return encoder_inputs, decoder_inputs, decoder_outputs

    def find_indices_srnn( self, data, action, original_format=False):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState( SEED )

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
        T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        return idx

    def get_batch_srnn(self, action):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses
          action: the action to load data from
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """
        data = self.test_set
        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                  "posing", "purchases", "sitting", "sittingdown", "smoking",
                  "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if not action in actions:
          raise ValueError("Unrecognized action {0}".format(action))

        frames = {}
        frames[ action ] = self.find_indices_srnn( data, action )

        batch_size = 8 # we always evaluate 8 seeds
        subject    = 5 # we always evaluate on subject 5
        seq_length_in = self.seq_length_in
        seq_length_out = self.seq_length_out

        seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

        encoder_inputs  = np.zeros((batch_size, self.num_joint, self.encoder_input_size), dtype=float)
        decoder_inputs  = np.zeros((batch_size, self.seq_length_out, self.num_joint, self.decoder_input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, self.seq_length_out, self.decoder_output_size), dtype=float)

        # Compute the number of frames needed
        total_frames = self.seq_length_in + self.seq_length_out

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in range( batch_size ):

          _, subsequence, idx = seeds[i]
          idx = idx + 50

          data_sel = data[ (subject, action, subsequence, 'even') ]

          data_sel = data_sel[(idx-seq_length_in):(idx+seq_length_out) ,:]

          # add transform to coordinate

          data_to_transform = data_sel
          # get the class info
          if self.one_hot:
              class_encoding = data_sel[0,-len(self.actions):]
              data_to_transform = data_sel[:,:-len(self.actions)]
              data_sel = data_sel[:,:-len(self.actions)]
              encoder_inputs[i,:,-len(self.actions):] = class_encoding
              decoder_inputs[i,:,:,-len(self.actions):] = class_encoding
          # do the transfrom
          encoder_input = np.reshape(
            data_to_transform[0:self.seq_length_in,:], [-1, self.num_joint, 3])
          encoder_input =  np.reshape(np.transpose(encoder_input, [1,0,2]),[self.num_joint,-1])
          decoder_input = np.reshape(
            data_to_transform[self.seq_length_in-1:total_frames-1, :],
            [-1, self.num_joint, 3]
          )
          decoder_output = data_to_transform[self.seq_length_in:, :]
          # Add the data
          if self.one_hot:
              encoder_inputs[i,:,0:-len(self.actions)]  = encoder_input
              decoder_inputs[i,:,:,0:-len(self.actions)]  = decoder_input
          else:
              encoder_inputs[i]  = encoder_input
              decoder_inputs[i]  = decoder_inp
          decoder_outputs[i] = decoder_output

        encoder_inputs = torch.tensor(encoder_inputs,dtype=self.dtype).to(self.device)
        decoder_inputs = torch.tensor(decoder_inputs,dtype=self.dtype).permute(1,0,2,3).to(self.device)
        decoder_outputs = torch.tensor(decoder_outputs,dtype=self.dtype).permute(1,0,2).to(self.device)

        return encoder_inputs, decoder_inputs, decoder_outputs


    def get_srnn_gts(self, to_euler=True):
      """
      Get the ground truths for srnn's sequences, and convert to Euler angles.
      (the error is always computed in Euler angles).

      Args
        actions: a list of actions to get ground truths for.
        model: training model we are using (we only use the "get_batch" method).
        test_set: dictionary with normalized training data.
        data_mean: d-long vector with the mean of the training data.
        data_std: d-long vector with the standard deviation of the training data.
        dim_to_ignore: dimensions that we are not using to train/predict.
        one_hot: whether the data comes with one-hot encoding indicating action.
        to_euler: whether to convert the angles to Euler format or keep thm in exponential map

      Returns
        srnn_gts_euler: a dictionary where the keys are actions, and the values
          are the ground_truth, denormalized expected outputs of srnns's seeds.
      """
      srnn_gts_euler = {}
      for action in self.actions:

        srnn_gt_euler = []
        _, _, srnn_expmap = self.get_batch_srnn(action)
        srnn_expmap = srnn_expmap.permute(1,0,2).cpu().detach().numpy()
        # expmap -> rotmat -> euler
        for i in np.arange(srnn_expmap.shape[0] ):
          denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], self.data_mean, self.data_std, self.dim_to_ignore, self.actions, self.one_hot )

          if to_euler:
            for j in np.arange( denormed.shape[0] ):
              for k in np.arange(3,97,3):
                denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3]))

          srnn_gt_euler.append( denormed );

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler
      return srnn_gts_euler

    def define_actions(self, action):
      """
      Define the list of actions we are using.

      Args
        action: String with the passed action. Could be "all"
      Returns
        actions: List of strings of actions
      Raises
        ValueError if the action is not included in H3.6M
      """

      actions = ["walking", "eating", "smoking", "discussion",  "directions",
                  "greeting", "phoning", "posing", "purchases", "sitting",
                  "sittingdown", "takingphoto", "waiting", "walkingdog",
                  "walkingtogether"]

      if action in actions:
        return [action]

      if action == "all":
        return actions

      if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

      raise( ValueError, "Unrecognized action: %d" % action )

    def read_all_data(self,actions, seq_length_in, seq_length_out, data_dir,        one_hot,use_GNN=False):
      """
      Loads data for training/testing and normalizes it.

      Args
        actions: list of strings (actions) to load
        seq_length_in: number of frames to use in the burn-in sequence
        seq_length_out: number of frames to use in the output sequence
        data_dir: directory to load the data from
        one_hot: whether to use one-hot encoding per action
      Returns
        train_set: dictionary with normalized training data
        test_set: dictionary with test data
        data_mean: d-long vector with the mean of the training data
        data_std: d-long vector with the standard dev of the training data
        dim_to_ignore: dimensions that are not used becaused stdev is too small
        dim_to_use: dimensions that we are actually using in the model
      """

      # === Read training data ===
      print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
               seq_length_in, seq_length_out))

      # train_subject_ids = [1,6,7,8,9,11]
      train_subject_ids = [1]
      test_subject_ids = [5]

      train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
      test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

      # Compute normalization stats
      data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train, use_GNN)

      # Normalize -- subtract mean, divide by stdev
      train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot)
      test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot)
      print("done reading data.")
      return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

if __name__ == "__main__":
    dataset = HumanDataset(
        action='all',
        seq_length_in=50,
        seq_length_out=25,
        encoder_input_size=150,
        decoder_input_size=3,
        decoder_output_size=63,
        num_joint=21,
        data_dir='data/h3.6m/dataset/',
        one_hot=True,
        use_GNN=True,
        device="cuda"
    )

    enc_in,dec_in,dec_out = dataset.get_batch_train(batch_size=8)
    print(enc_in.shape)
    print(dec_in.shape)
    print(dec_out.shape)
    enc_in,dec_in,dec_out = dataset.get_batch_validation()
    print(enc_in.shape)
    print(dec_in.shape)
    print(dec_out.shape)
    enc_in, dec_in,dec_out = dataset.get_batch_srnn(action="walking")
    print(enc_in.shape)
    print(dec_in.shape)
    print(dec_out.shape)
