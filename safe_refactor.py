import os
import re

# Mapping for the new structure
mapping = {
    r'\bfrom analysis\b': 'from quantum.domain.analysis',
    r'\bimport analysis\b': 'import quantum.domain.analysis',
    r'\bfrom core\b': 'from quantum.domain.core',
    r'\bimport core\b': 'import quantum.domain.core',
    r'\bfrom data\b': 'from quantum.domain.data',
    r'\bimport data\b': 'import quantum.domain.data',
    r'\bfrom ml\b': 'from quantum.domain.ml',
    r'\bimport ml\b': 'import quantum.domain.ml',
    r'\bfrom risk\b': 'from quantum.domain.risk',
    r'\bimport risk\b': 'import quantum.domain.risk',
    r'\bfrom strategies\b': 'from quantum.domain.strategies',
    r'\bimport strategies\b': 'import quantum.domain.strategies',
    r'\bfrom backtest\b': 'from quantum.application.backtest',
    r'\bimport backtest\b': 'import quantum.application.backtest',
    r'\bfrom reporting\b': 'from quantum.application.reporting',
    r'\bimport reporting\b': 'import quantum.application.reporting',
    r'\bfrom api\b': 'from quantum.infrastructure.api',
    r'\bimport api\b': 'import quantum.infrastructure.api',
    r'\bfrom db\b': 'from quantum.infrastructure.db',
    r'\bimport db\b': 'import quantum.infrastructure.db',
    r'\bfrom config\b': 'from quantum.shared.config',
    r'\bimport config\b': 'import quantum.shared.config',
    r'\bfrom utils\b': 'from quantum.shared.utils',
    r'\bimport utils\b': 'import quantum.shared.utils',
    r'\bfrom web3_innovation\b': 'from quantum.shared.web3',
    r'\bimport web3_innovation\b': 'import quantum.shared.web3',
}

# Special mapping for the config regression
config_fix = (r'from quantum\.shared\.config\.settings import quantum\.shared\.config', 'from quantum.shared.config.settings import config')

def refactor(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                
                # Robust read
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                
                # Apply structural mapping
                for pattern, replacement in mapping.items():
                    new_content = re.sub(pattern, replacement, new_content)
                
                # Apply config fix
                new_content = re.sub(config_fix[0], config_fix[1], new_content)
                
                # Final check for sys.path hacks that might break local imports
                # (Removing the sys.path.insert lines that were tailored for root)
                new_content = re.sub(r'sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)', '', new_content)
                new_content = re.sub(r'sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)', '', new_content)
                
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Refactored: {path}")

if __name__ == "__main__":
    # Refactor everything in src/quantum and tests
    refactor('src/quantum')
    refactor('tests')
    print("Refactoring complete.")
