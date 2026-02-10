# AI Swarm Detection

**Team:** Murali Ediga (muralikrish9)

## Problem Statement
Detection gap for multi-agent AI attack swarms in cybersecurity.

## Objectives
- Three-class classification (AI/Human/Bot)
- Behavioral fingerprinting
- Active countermeasures

## System Architecture
Four-layer architecture:
1. Data Collection
2. Feature Engineering
3. Detection
4. Countermeasures

## Datasets
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- Palisade Honeypot Logs

## References
- GraphIDS
- GUARDIAN
- Easy2Hard (NeurIPS 2025)

## Repository Structure
- `/proposal`: PDF proposal document
- `/docs`: System architecture diagrams, design notes, experiment logs
- `/src`: Source code organized by component (`detection/`, `countermeasures/`, `features/`)
- `/reproducibility`: Data pipelines, experiment scripts, evaluation procedures, environment setup
