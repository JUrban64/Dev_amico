from transformers import AutoTokenizer, EsmModel
import torch
import numpy as np
import os



class ESMFeatureExtractor:
    """
    Extract ESM-2 embeddings for protein sequences
    """
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """
        Options:
        - esm2_t33_650M_UR50D (650M params, 1280D embeddings)
        - esm2_t36_3B_UR50D (3B params, 2560D embeddings) - nejlepší
        - esm2_t30_150M_UR50D (150M params, 640D embeddings) - rychlejší
        """
        print(f"Loading ESM-2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available (CUDA or MPS)
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def extract_embeddings(self, sequence):
        """
        Extract per-residue embeddings
        
        Args:
            sequence: amino acid sequence (string)
        
        Returns:
            embeddings: [L, 1280] numpy array
        """
        if not sequence or len(sequence) == 0:
            raise ValueError("Empty sequence provided to extract_embeddings")
        
        # Tokenize – přidáno truncation pro dlouhé sekvence
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt",
            add_special_tokens=True,  # Adds <cls> and <eos>
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings (remove <cls> and <eos> tokens)
        embeddings = outputs.last_hidden_state[0, 1:-1, :]  # [L, 1280]
        
        return embeddings.cpu().numpy()
    
    def extract_binding_site_embeddings(self, full_sequence, bs_indices):
        """
        Extract embeddings only for binding site residues
        
        Args:
            full_sequence: full protein sequence
            bs_indices: list of binding site residue indices
        
        Returns:
            bs_embeddings: [n_bs, 1280]
            valid_indices: list of indices that were actually used
                           (after truncation filtering)
        """
        # Get full embeddings (může být oříznuté na 1022 residues)
        full_embeddings = self.extract_embeddings(full_sequence)
        
        # Filtruj indexy, které jsou mimo rozsah embeddingů (po truncation)
        max_idx = full_embeddings.shape[0]
        valid_indices = [i for i in bs_indices if i < max_idx]
        
        if len(valid_indices) == 0:
            raise ValueError(
                f"No valid binding site indices after truncation. "
                f"Embedding length: {max_idx}, "
                f"BS indices range: {min(bs_indices)}-{max(bs_indices)}"
            )
        
        if len(valid_indices) < len(bs_indices):
            print(f"  ⚠ {len(bs_indices) - len(valid_indices)} BS residues "
                  f"mimo rozsah embeddingů (seq truncated to {max_idx})")
        
        # Select binding site residues
        bs_embeddings = full_embeddings[valid_indices, :]
        
        return bs_embeddings, valid_indices
    
    def batch_extract(self, sequences, batch_size=8):
        """
        Extract embeddings for multiple sequences (more efficient)
        """
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract (handle padding)
            for j, seq in enumerate(batch):
                seq_len = len(seq)
                emb = outputs.last_hidden_state[j, 1:seq_len+1, :]
                all_embeddings.append(emb.cpu().numpy())
        
        return all_embeddings

    def extract_and_save_to_disk(self, sequences, output_dir, 
                                  max_length=1024, batch_size=1):
        """
        Extrahuje ESM embeddingy a ukládá je inkrementálně na disk
        jako jednotlivé .npy soubory. Přeskakuje sekvence, pro které
        soubor již existuje (resume-safe).
        
        Toto je hlavní mechanismus pro úsporu RAM – embeddingy se
        nikdy nehromadí v paměti.
        
        Args:
            sequences: list of (seq_id, sequence) tuples
            output_dir: složka pro .npy soubory
            max_length: maximální délka sekvence (ESM truncation)
            batch_size: počet sekvencí najednou (1 = nejmenší RAM)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        total = len(sequences)
        skipped = 0
        computed = 0
        
        for i, (seq_id, seq) in enumerate(sequences):
            npy_path = os.path.join(output_dir, f"{seq_id}.npy")
            
            # Přeskoč pokud již existuje (resume-safe)
            if os.path.exists(npy_path):
                skipped += 1
                continue
            
            truncated = seq[:max_length]
            emb = self.extract_embeddings(truncated)  # [L, 1280] numpy
            np.save(npy_path, emb.astype(np.float16))
            computed += 1
            
            if (i + 1) % 100 == 0:
                print(f"    [{i+1}/{total}] uloženo {computed}, "
                      f"přeskočeno {skipped}")
        
        print(f"  ✓ Hotovo: {computed} nových, {skipped} přeskočeno "
              f"z {total} celkem")


# Použití
if __name__ == '__main__':
    esm_extractor = ESMFeatureExtractor()

    # Pre-compute embeddings pro všechny binding sites
    for bs_info in binding_sites:
        bs_embeddings = esm_extractor.extract_binding_site_embeddings(
            bs_info['full_sequence'],
            bs_info['binding_site_indices']
        )
        bs_info['esm_embeddings'] = bs_embeddings
        print(f"Extracted embeddings: {bs_embeddings.shape}")