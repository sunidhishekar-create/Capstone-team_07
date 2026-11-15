# &#x20;

# 

# \## Scope of Our Implementation

# 

# This project implements the \*\*original DRAG algorithm\*\* for federated learning. BR-DRAG is not implemented in this work.

# 

# The implementation evaluates DRAG on three real datasets:

# \- \*\*EMNIST Balanced\*\*

# \- \*\*CIFAR-10\*\*

# \- \*\*CIFAR-100\*\*

# 

# The experiments compare two data-partition settings, β = 0.1 and β = 0.5, to study the effect of non-IID client data distributions.

# 

# \## Experimental Configuration

# 

# | Parameter | Value |

# |---|---|

# | Number of clients (M) | 40 |

# | Clients selected per round (S) | 10 |

# | Local SGD steps (U) | 5 |

# | Batch size | 10 |

# | Learning rate | 0.01 |

# | α | 0.25 |

# | c for β = 0.1 | 0.25 |

# | c for β = 0.5 | 0.1 |

# | EMNIST rounds | 3000 |

# | CIFAR-10 and CIFAR-100 rounds | 4000 |

# 

# \## Final Experimental Results

# 

# | Dataset | β | Rounds | Test Accuracy | Test Loss | Mean λ |

# |---|---:|---:|---:|---:|---:|

# | EMNIST Balanced | 0.1 | 3000 | 87.16% | 0.4687 | 0.2566 |

# | EMNIST Balanced | 0.5 | 3000 | 88.07% | 0.3636 | 0.1023 |

# | CIFAR-10 | 0.1 | 4000 | 74.59% | 1.6749 | 0.2372 |

# | CIFAR-10 | 0.5 | 4000 | 73.53% | 1.5299 | 0.1115 |

# | CIFAR-100 | 0.1 | 4000 | 45.26% | 6.7822 | 0.2465 |

# | CIFAR-100 | 0.5 | 4000 | 40.97% | 5.7664 | 0.0989 |

# 

# \## Results and Observations

# 

# \- EMNIST achieved its highest observed test accuracy of \*\*88.07%\*\* with β = 0.5.

# \- CIFAR-10 achieved its highest observed test accuracy of \*\*74.59%\*\* with β = 0.1.

# \- CIFAR-100 achieved its highest observed test accuracy of \*\*45.26%\*\* with β = 0.1.

# \- The mean λ was lower for β = 0.5 across all three datasets in these runs.

# \- The results show that the effect of the data-partition setting differs across datasets.

# 

# \## Limitations

# 

# Each configuration was evaluated with one seed. Therefore, these results are preliminary and do not establish statistical significance or guarantee exact reproduction of every setting in the original paper.

# 

# \## Detailed Documentation and Results

# 

# The DRAG algorithm explanation is documented in \[`docs/drag\_algorithm.md`](docs/drag\_algorithm.md).

# 

# Experiment plots, summaries, and checkpoints are stored separately in Google Drive under `Capstone/DRAG\_Project/`.



