import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

function load (component) {
  return () => System.import(`@/${component}.vue`)
}

const router = new Router({
  routes: [
    {
      path: '/',
      component: load('views/Dash'),
      children: [
        {
          path: 'patients',
          name: 'patients',
          component: load('views/Patients')
        },
        {
          path: 'patients/:patientID',
          name: 'single-patient',
          component: load('views/SinglePatient')
        },
        {
          path: 'patients/:patientID/:versionNumber',
          name: 'single-versioned-patient',
          component: load('views/SingleVersionedPatient')
        },
        {
          path: 'medications',
          name: 'medications',
          component: load('views/Medications')
        },
        {
          path: 'medications/:medicationID',
          name: 'single-medication',
          component: load('views/SingleMedication')
        },
        {
          path: 'medications/:medicationID/:versionNumber',
          name: 'single-versioned-medication',
          component: load('views/SingleVersionedMedication')
        },
        {
          path: 'statements',
          name: 'statements',
          component: load('views/Statements')
        },
        {
          path: 'statements/:statementID',
          name: 'single-statement',
          component: load('views/SingleStatement')
        },
        {
          path: 'statements/:statementID/:versionNumber',
          name: 'single-versioned-statement',
          component: load('views/SingleVersionedStatement')
        },
        {
          path: 'observations',
          name: 'observations',
          component: load('views/Observations')
        },
        {
          path: 'observations/:observationID',
          name: 'single-observation',
          component: load('views/SingleObservation')
        }
      ]
    },
    {
      path: '*',
      component: load('views/NotFound')
    }
  ]
})

export default router
