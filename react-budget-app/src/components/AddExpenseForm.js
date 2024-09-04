import React,{useState,useContext} from 'react'
import {AppContext} from '../contexts/AppContext'

const AddExpenseForm = () => {

  const [name,setName] = useState('');
  const [cost,setCost] =useState(0);

  const { dispatch }=useContext(AppContext);

  const handleSubmit =(e)=> {
      e.preventDefault();
  

  const expense ={
    id: crypto.randomUUID(),
    name:name,
    cost:cost
  };

  dispatch({
    type:'ADD_EXPENSE',
    payload:expense
  })
  setName('')
  setCost(0)

};

  return (
    <form onSubmit={handleSubmit}>
      <div className='row'>
        <div className='col-sm col-lg-4'>
          <label>지출</label>
          <input 
            required='required' 
            type='text'
            className='form-control'
            id = 'name'
            value = {name}
            onChange ={(e)=> setName(e.target.value)}
          />
          </div>
          <div>
            <label>비용</label>
            <input
              required="required"
              type="number"
              className="form-control"
              id ="cost"
              value={cost} 
              onChange = {(e)=> setCost(e.target.valueAsNumber)}
            />

          </div>
          <div className='row mt-3'>
          <div className='col-sm'>
            <button type='submit' className='btn btn-primary'>
                추가
              </button>

            </div>

          </div>
        </div>
    </form>
  );
};

export default AddExpenseForm