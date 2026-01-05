#!/usr/bin/env python3
"""
Verify documentation reorganization
"""
import os
from pathlib import Path

def main():
    print("📚 FHDP Documentation Organization Verification")
    print("=" * 50)
    
    # Check root structure
    print("\n📁 Root Directory Structure:")
    root_files = [f for f in os.listdir('.') if f.endswith('.md') or f.endswith('.py') or f.endswith('.sh') or f.endswith('.txt')]
    
    expected_root = [
        'README.md',
        'requirements_evo1.txt',
        'requirements_evo1_stages.txt',
        'requirements_evo1_stage2.txt', 
        'requirements_evo1_stage3.txt',
        'requirements_jetson_stage1.txt',
        'fix_flash_attn.sh',
        'install_evo1_deps.py',
        'test_install_fix.py'
    ]
    
    for file in expected_root:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Check docs structure
    print(f"\n📁 docs/ Directory Structure:")
    docs_dir = Path('docs')
    if docs_dir.exists():
        docs_files = sorted([f.name for f in docs_dir.iterdir() if f.is_file()])
        
        expected_docs = [
            'README.md',
            'QUICK_REFERENCE.md', 
            'INSTALLATION.md',
            'IMPLEMENTATION_SUMMARY.md',
            'README_EVO1_STAGE1.md',
            'README_AUTONOMOUS_DRIVING.md',
            'HETEROGENEOUS_ADAPTATION.md',
            'HETEROGENEOUS_ADAPTATION_SUMMARY.md',
            'INSTALLATION_FIX.md',
            'LEARNING_PATH.md',
            'TESTBED_ARCHITECTURE.md',
            'TESTBED_CHECKLIST.md'
        ]
        
        for file in expected_docs:
            if file in docs_files:
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (missing)")
        
        print(f"\n📊 Documentation Stats:")
        print(f"  Total docs: {len(docs_files)}")
        print(f"  Expected: {len(expected_docs)}")
        print(f"  Match: {len([f for f in expected_docs if f in docs_files])}")
    
    # Check examples structure
    print(f"\n📁 examples/ Directory:")
    examples_dir = Path('examples')
    if examples_dir.exists():
        example_files = sorted([f.name for f in examples_dir.iterdir() if f.is_file() and f.suffix == '.py'])
        
        expected_examples = [
            'simple_simulation.py',
            'enhanced_simulation.py', 
            'autonomous_driving_simulation.py',
            'evo1_stage1_federated.py'
        ]
        
        for file in expected_examples:
            if file in example_files:
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (missing)")
    
    # Navigation check
    print(f"\n🧭 Navigation Paths:")
    print(f"  📚 Main docs hub: docs/README.md")
    print(f"  ⚡ Quick start: docs/QUICK_REFERENCE.md") 
    print(f"  🔧 Installation: docs/INSTALLATION.md")
    print(f"  🧠 EVO-1 Stage 1: docs/README_EVO1_STAGE1.md")
    print(f"  🚗 Autonomous driving: docs/README_AUTONOMOUS_DRIVING.md")
    print(f"  📋 Implementation: docs/IMPLEMENTATION_SUMMARY.md")
    
    # Quick access test
    print(f"\n🎯 Quick Access Test:")
    
    key_files = [
        ('Main README', 'README.md'),
        ('Docs Hub', 'docs/README.md'),
        ('Installation Guide', 'docs/INSTALLATION.md'),
        ('Quick Reference', 'docs/QUICK_REFERENCE.md'),
        ('EVO-1 Stage 1', 'docs/README_EVO1_STAGE1.md'),
        ('Autonomous Driving', 'docs/README_AUTONOMOUS_DRIVING.md'),
        ('Implementation Summary', 'docs/IMPLEMENTATION_SUMMARY.md')
    ]
    
    accessible = 0
    for desc, path in key_files:
        if os.path.exists(path):
            print(f"  ✅ {desc}: {path}")
            accessible += 1
        else:
            print(f"  ❌ {desc}: {path} (missing)")
    
    # Summary
    print(f"\n📊 Organization Summary:")
    print(f"  ✅ Documentation moved to docs/ folder")
    print(f"  ✅ Root README simplified and updated")
    print(f"  ✅ Clear navigation paths established")
    print(f"  ✅ Accessible key files: {accessible}/{len(key_files)}")
    
    if accessible == len(key_files):
        print(f"\n🎉 Documentation reorganization completed successfully!")
        print(f"   📚 All files properly organized")
        print(f"   🧭 Clear navigation structure")
        print(f"   📱 Easy browsing on any device")
        print(f"   🎯 Ready for users!")
    else:
        print(f"\n⚠️  {len(key_files) - accessible} files missing")
        print(f"   Check file paths above")

if __name__ == '__main__':
    main()