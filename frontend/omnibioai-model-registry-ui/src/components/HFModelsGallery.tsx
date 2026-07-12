import { useMemo, useState, type CSSProperties } from "react";

type HFModel = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  downloads: string;
};

type HFCategory = {
  category: string;
  models: HFModel[];
};

const HF_BIO_MODELS: HFCategory[] = [
  {
    category: "DNA & Genomics",
    models: [
      {
        id: "facebook/esm2_t33_650M_UR50D",
        name: "ESM2 650M",
        description: "Protein language model trained on UniRef50",
        tags: ["protein", "embeddings", "genomics"],
        downloads: "500K+",
      },
      {
        id: "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        name: "Nucleotide Transformer 500M",
        description: "DNA foundation model trained on the human reference genome",
        tags: ["DNA", "genomics", "foundation-model"],
        downloads: "50K+",
      },
      {
        id: "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        name: "Nucleotide Transformer 2.5B Multi-Species",
        description: "DNA foundation model trained across multiple species",
        tags: ["DNA", "genomics", "multi-species"],
        downloads: "20K+",
      },
      {
        id: "zhihan1996/DNABERT-2-117M",
        name: "DNABERT-2",
        description: "Multi-species DNA language model",
        tags: ["DNA", "BERT", "multi-species"],
        downloads: "30K+",
      },
      {
        id: "kuleshov-group/hyenadna-large-1m-seqlen",
        name: "HyenaDNA Large",
        description: "Long-range genomic sequence model (1M context)",
        tags: ["DNA", "long-range", "genomics"],
        downloads: "20K+",
      },
    ],
  },
  {
    category: "Protein Structure",
    models: [
      {
        id: "facebook/esmfold_v1",
        name: "ESMFold v1",
        description: "End-to-end protein structure prediction",
        tags: ["protein", "structure", "folding"],
        downloads: "200K+",
      },
      {
        id: "Rostlab/prot_t5_xl_uniref50",
        name: "ProtT5-XL",
        description: "Protein language model based on T5",
        tags: ["protein", "T5", "embeddings"],
        downloads: "100K+",
      },
      {
        id: "Rostlab/prot_bert",
        name: "ProtBERT",
        description: "BERT model trained on protein sequences",
        tags: ["protein", "BERT", "embeddings"],
        downloads: "80K+",
      },
      {
        id: "ElnaggarLab/ankh-large",
        name: "ANKH Large",
        description: "Efficient protein language model",
        tags: ["protein", "efficient", "embeddings"],
        downloads: "15K+",
      },
    ],
  },
  {
    category: "Single Cell",
    models: [
      {
        id: "ctheodoris/Geneformer",
        name: "Geneformer",
        description: "Context-aware single-cell transformer",
        tags: ["scRNA-seq", "transformer", "single-cell"],
        downloads: "80K+",
      },
      {
        id: "bowang-lab/scGPT_human",
        name: "scGPT Human",
        description: "Foundation model for single-cell biology",
        tags: ["scRNA-seq", "GPT", "human"],
        downloads: "40K+",
      },
    ],
  },
  {
    category: "Clinical NLP",
    models: [
      {
        id: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        name: "BiomedBERT",
        description: "BERT trained on PubMed abstracts",
        tags: ["NLP", "clinical", "BERT"],
        downloads: "1M+",
      },
      {
        id: "dmis-lab/biobert-v1.1",
        name: "BioBERT v1.1",
        description: "Biomedical language model",
        tags: ["NLP", "biomedical", "BERT"],
        downloads: "2M+",
      },
      {
        id: "allenai/biomed_roberta_base",
        name: "BioMed RoBERTa",
        description: "RoBERTa trained on biomedical literature",
        tags: ["NLP", "RoBERTa", "biomedical"],
        downloads: "500K+",
      },
      {
        id: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        name: "SapBERT",
        description: "Biomedical entity representation model",
        tags: ["NLP", "entity-linking", "biomedical"],
        downloads: "50K+",
      },
    ],
  },
  {
    category: "Drug Discovery",
    models: [
      {
        id: "seyonec/ChemBERTa-zinc-base-v1",
        name: "ChemBERTa",
        description: "Chemical language model for drug discovery",
        tags: ["chemistry", "SMILES", "drug-discovery"],
        downloads: "100K+",
      },
      {
        id: "DeepChem/ChemBERTa-77M-MTR",
        name: "ChemBERTa-77M-MTR",
        description: "Multi-task regression chemical language model",
        tags: ["chemistry", "SMILES", "regression"],
        downloads: "30K+",
      },
      {
        id: "ncfrey/ChemGPT-1.2B",
        name: "ChemGPT 1.2B",
        description: "GPT model for molecular generation",
        tags: ["chemistry", "generation", "GPT"],
        downloads: "20K+",
      },
    ],
  },
  {
    category: "Pathology & Imaging",
    models: [
      {
        id: "paige-ai/Virchow",
        name: "Virchow",
        description: "Foundation model for computational pathology",
        tags: ["pathology", "imaging", "foundation"],
        downloads: "10K+",
      },
      {
        id: "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        name: "BiomedCLIP",
        description: "CLIP model for biomedical images and text",
        tags: ["multimodal", "CLIP", "imaging"],
        downloads: "30K+",
      },
    ],
  },
];

function hfUrl(id: string): string {
  return `https://huggingface.co/${id}`;
}

export default function HFModelsGallery() {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [search, setSearch] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const categories = useMemo(() => ["All", ...HF_BIO_MODELS.map((c) => c.category)], []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return HF_BIO_MODELS.filter((cat) => selectedCategory === "All" || cat.category === selectedCategory)
      .map((cat) => ({
        ...cat,
        models: cat.models.filter(
          (m) =>
            !q ||
            m.name.toLowerCase().includes(q) ||
            m.description.toLowerCase().includes(q) ||
            m.id.toLowerCase().includes(q) ||
            m.tags.some((t) => t.toLowerCase().includes(q))
        ),
      }))
      .filter((cat) => cat.models.length > 0);
  }, [selectedCategory, search]);

  async function copyId(id: string) {
    try {
      await navigator.clipboard.writeText(id);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId((cur) => (cur === id ? null : cur)), 1500);
    } catch {
      // clipboard unavailable — silently ignore
    }
  }

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 20 }}>🤗</span>
        <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>HuggingFace Bioinformatics Models</h2>
      </div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16 }}>
        Curated models for genomics, proteomics, single-cell, clinical NLP, drug discovery, and pathology.
      </div>

      <input
        type="text"
        placeholder="Search models, tags, or descriptions…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{ width: "100%", marginBottom: 12 }}
      />

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            style={{
              padding: "4px 12px",
              borderRadius: 20,
              border: "1px solid",
              borderColor: selectedCategory === cat ? "var(--accent)" : "var(--border)",
              background: selectedCategory === cat ? "var(--accent-bg)" : "transparent",
              color: selectedCategory === cat ? "var(--accent)" : "var(--text-muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            {cat}
          </button>
        ))}
      </div>

      {filtered.length === 0 && (
        <div style={{ color: "var(--text-muted)", textAlign: "center", paddingTop: 40 }}>
          No models match "{search}".
        </div>
      )}

      {filtered.map((cat) => (
        <div key={cat.category} style={{ marginBottom: 28 }}>
          <h3
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: "var(--accent)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 10,
            }}
          >
            {cat.category}
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
            {cat.models.map((model) => (
              <div key={model.id} style={cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8, marginBottom: 6 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>{model.name}</h4>
                  <span style={{ fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}>↓ {model.downloads}</span>
                </div>

                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8, fontFamily: "monospace" }}>{model.id}</div>

                <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 10, lineHeight: 1.5 }}>{model.description}</p>

                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
                  {model.tags.map((tag) => (
                    <span key={tag} style={tagStyle}>
                      {tag}
                    </span>
                  ))}
                </div>

                <div style={{ display: "flex", gap: 8 }}>
                  <a
                    href={hfUrl(model.id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      flex: 1,
                      textAlign: "center",
                      padding: "5px 0",
                      borderRadius: 4,
                      border: "1px solid var(--accent)",
                      color: "var(--accent)",
                      background: "var(--accent-bg)",
                      fontSize: 12,
                      textDecoration: "none",
                    }}
                  >
                    🤗 View on HF
                  </a>
                  <button style={{ flex: 1, fontSize: 12 }} onClick={() => copyId(model.id)}>
                    {copiedId === model.id ? "Copied ✓" : "📋 Copy ID"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

const cardStyle: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  padding: 14,
};

const tagStyle: CSSProperties = {
  fontSize: 10,
  padding: "2px 7px",
  background: "var(--surface-2)",
  borderRadius: 4,
  color: "var(--text-muted)",
};
