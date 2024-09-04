import React,{useContext, useState} from 'react';
import { AppContext } from '../contexts/AppContext';
import EditBudget from './EditBudget';
import ViewBudget from './ViewBudget';

const Budget = () => {

    const {budget,dispatch} = useContext(AppContext);

    const[isEditing,setIsEditing] = useState(false)

  
  
    return (
    <div className ='
    alert alert-primary p-3 d-flex align-items-center
    justify-content-between
    '>
      {isEditing ? 
      <EditBudget dispatch={dispatch} setIsEditing={setIsEditing} budget ={budget}/>
      :
        <ViewBudget setIsEditing={setIsEditing} budget ={budget} />

    }
    </div>
  )
}

export default Budget