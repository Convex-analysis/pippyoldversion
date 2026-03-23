#!/bin/bash
#
# Quick runner for Template Manager tests
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_banner() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Template Manager Test Suite                         ║"
    echo "║     Template Manager 测试套件                             ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo "Usage: $0 [test|demo|all] [options]"
    echo ""
    echo "Commands:"
    echo "  test        - Run unit tests"
    echo "  demo        - Run interactive demo"
    echo "  all         - Run tests and demos"
    echo "  help        - Show this help"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    - Verbose output"
    echo "  -k PATTERN       - Run tests matching pattern"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 demo"
    echo "  $0 test -k partitioning"
    echo "  $0 all"
}

# Set Python path
setup_env() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/.."
}

# Run unit tests
run_tests() {
    local pattern=""
    local verbose=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -k|--pattern)
                pattern="-k $2"
                shift 2
                ;;
            -v|--verbose)
                verbose="-v"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    echo -e "${BLUE}Running Template Manager Tests...${NC}"
    echo ""
    
    python test_template_manager.py $pattern $verbose
}

# Run demo
run_demo() {
    echo -e "${BLUE}Running Template Partitioning Demo...${NC}"
    echo ""
    
    python demo_template_partitioning.py
}

# Run all
run_all() {
    print_banner
    
    echo -e "${YELLOW}Step 1: Running Unit Tests${NC}"
    echo ""
    run_tests "$@"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Tests passed!${NC}"
    else
        echo -e "${YELLOW}⚠ Some tests failed${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Step 2: Running Demo${NC}"
    echo ""
    run_demo
    
    echo ""
    echo -e "${GREEN}✓ All operations completed!${NC}"
}

# Main
main() {
    setup_env
    
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    case $1 in
        test)
            shift
            run_tests "$@"
            ;;
        demo)
            shift
            run_demo "$@"
            ;;
        all)
            shift
            run_all "$@"
            ;;
        help|--help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown command: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
