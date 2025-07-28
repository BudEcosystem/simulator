"""Hardware API routes."""
from flask import request, jsonify
from BudSimulator.src.hardware import BudHardware
from BudSimulator.src.hardware_recommendation import HardwareRecommendation


def create_hardware_routes(app):
    """Create hardware-related routes."""
    
    @app.route('/api/hardware', methods=['POST'])
    def add_hardware():
        """Add new hardware."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            hardware = BudHardware()
            hardware.add_hardware(data)
            
            return jsonify({
                'success': True,
                'message': 'Hardware added successfully'
            }), 201
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Internal error: {str(e)}'
            }), 500
    
    @app.route('/api/hardware', methods=['GET'])
    def list_hardware():
        """List all hardware with optional filters."""
        try:
            hardware = BudHardware()
            
            # Get query parameters
            params = {
                'type': request.args.get('type'),
                'manufacturer': request.args.get('manufacturer'),
                'min_flops': float(request.args.get('min_flops')) if request.args.get('min_flops') else None,
                'max_flops': float(request.args.get('max_flops')) if request.args.get('max_flops') else None,
                'min_memory': float(request.args.get('min_memory')) if request.args.get('min_memory') else None,
                'max_memory': float(request.args.get('max_memory')) if request.args.get('max_memory') else None,
                'min_power': float(request.args.get('min_power')) if request.args.get('min_power') else None,
                'max_power': float(request.args.get('max_power')) if request.args.get('max_power') else None,
                'min_price': float(request.args.get('min_price')) if request.args.get('min_price') else None,
                'max_price': float(request.args.get('max_price')) if request.args.get('max_price') else None,
                'sort_by': request.args.get('sort_by', 'name'),
                'sort_order': request.args.get('sort_order', 'asc'),
                'limit': int(request.args.get('limit')) if request.args.get('limit') else None,
                'offset': int(request.args.get('offset', 0))
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            results = hardware.search_hardware(**params)
            
            return jsonify({
                'success': True,
                'hardware': results,
                'count': len(results)
            }), 200
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/hardware/filter', methods=['GET'])
    def filter_hardware():
        """Filter hardware (alias for list with filters)."""
        return list_hardware()
    
    @app.route('/api/hardware/recommend', methods=['POST'])
    def recommend_hardware():
        """Recommend hardware based on memory requirements."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Validate required field
            if 'total_memory_gb' not in data:
                return jsonify({
                    'success': False,
                    'error': 'total_memory_gb is required'
                }), 400
            
            total_memory_gb = float(data['total_memory_gb'])
            model_params_b = float(data.get('model_params_b')) if data.get('model_params_b') else None
            
            recommender = HardwareRecommendation()
            recommendations = recommender.recommend_hardware(total_memory_gb, model_params_b)
            
            # Return the enhanced structure directly
            return jsonify(recommendations), 200
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Internal error: {str(e)}'
            }), 500 